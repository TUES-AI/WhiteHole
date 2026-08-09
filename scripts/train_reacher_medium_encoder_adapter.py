"""Train a copied LeWM encoder for the Reacher medium visual shift."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from sklearn import preprocessing
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.visualize_reacher_shifts import render_variant


TRAIN_MODES = ("last_block", "last_25", "last_50", "last_75", "full")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a target encoder E_T for medium Reacher visuals while keeping "
            "the LeWM predictor frozen."
        )
    )
    parser.add_argument("--dataset-name", default="dmc/reacher_random")
    parser.add_argument("--policy", default="quentinll/lewm-reacher")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--output-dir",
        default="tmp_reacher_visualization/medium_encoder_adapter/full",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--val-samples", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--pair-weight", type=float, default=1.0)
    parser.add_argument("--dynamics-weight", type=float, default=1.0)
    parser.add_argument("--target-dynamics-weight", type=float, default=0.0)
    parser.add_argument("--metric-weight", type=float, default=0.0)
    parser.add_argument("--delta-metric-weight", type=float, default=0.0)
    parser.add_argument("--latent-swd-weight", type=float, default=0.0)
    parser.add_argument("--transition-swd-weight", type=float, default=0.0)
    parser.add_argument("--joint-swd-weight", type=float, default=0.0)
    parser.add_argument("--swd-projections", type=int, default=128)
    parser.add_argument("--identity-weight", type=float, default=0.05)
    parser.add_argument("--init-encoder-checkpoint", default=None)
    parser.add_argument("--train-mode", choices=TRAIN_MODES, default="full")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def resolve_cache_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    if os.environ.get("STABLEWM_HOME"):
        return Path(os.environ["STABLEWM_HOME"])
    return Path("stablewm_home")


def sample_transition_rows(
    dataset, count: int, seed: int, action_block: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, dict]:
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    action = dataset.get_col_data("action")

    rows = np.arange(len(step_idx) - action_block)
    valid = (
        (step_idx[rows] >= 3)
        & (step_idx[rows] <= 190 - action_block)
        & (ep_idx[rows + action_block] == ep_idx[rows])
        & (step_idx[rows + action_block] == step_idx[rows] + action_block)
    )
    candidates = rows[valid]

    action_offsets = np.arange(action_block)
    block_rows = candidates[:, None] + action_offsets[None, :]
    flat_actions = action[block_rows].reshape(len(candidates), -1)
    finite = np.isfinite(flat_actions).all(axis=1)
    candidates = candidates[finite]
    flat_actions = flat_actions[finite]

    if count > len(candidates):
        raise ValueError(
            f"Requested {count} transitions, but only {len(candidates)} are valid."
        )

    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(candidates), size=count, replace=False))
    start_rows = candidates[chosen]
    next_rows = start_rows + action_block
    action_blocks = flat_actions[chosen]
    return (
        start_rows,
        next_rows,
        action_blocks,
        dataset.get_row_data(start_rows),
        dataset.get_row_data(next_rows),
    )


def normalize_images(frames: np.ndarray) -> torch.Tensor:
    mean = torch.tensor(spt.data.dataset_stats.ImageNet["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(spt.data.dataset_stats.ImageNet["std"]).view(1, 3, 1, 1)
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    return (x - mean) / std


def render_medium_source(rows: dict) -> tuple[torch.Tensor, torch.Tensor]:
    states = {"qpos": rows["qpos"], "qvel": rows["qvel"]}
    source = np.stack(render_variant(states, "source", image_size=224))
    medium = np.stack(render_variant(states, "medium_visual", image_size=224))
    return normalize_images(source), normalize_images(medium)


def encode_batches(
    model,
    images: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    embs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size].to(device)
            out = model.encode({"pixels": batch.unsqueeze(1)})
            embs.append(out["emb"][:, 0].detach().cpu())
    return torch.cat(embs, dim=0)


def fit_action_scaler(dataset) -> preprocessing.StandardScaler:
    action_data = dataset.get_col_data("action")
    action_data = action_data[~np.isnan(action_data).any(axis=1)]
    return preprocessing.StandardScaler().fit(action_data)


def standardize_action_blocks(
    action_blocks: np.ndarray, scaler: preprocessing.StandardScaler, action_dim: int
) -> torch.Tensor:
    shape = action_blocks.shape
    actions = action_blocks.reshape(-1, action_dim)
    normalized = scaler.transform(actions).reshape(shape)
    return torch.from_numpy(normalized).float()


def predict_next(model, aligned_start: torch.Tensor, action_block: torch.Tensor):
    act_emb = model.action_encoder(action_block.unsqueeze(1))
    return model.predict(aligned_start.unsqueeze(1), act_emb)[:, -1]


def set_trainable_encoder_depth(model, mode: str) -> list[str]:
    if mode not in TRAIN_MODES:
        raise ValueError(f"Unknown train mode {mode!r}")

    model.requires_grad_(False)
    encoder = model.encoder
    modules = []
    names = []

    layers = list(encoder.encoder.layer)
    if mode == "full":
        modules.append(encoder)
        names.append("encoder")
    else:
        count_by_mode = {
            "last_block": 1,
            "last_25": max(1, len(layers) // 4),
            "last_50": max(1, len(layers) // 2),
            "last_75": max(1, (3 * len(layers)) // 4),
        }
        count = count_by_mode[mode]
        first_layer = len(layers) - count
        for layer_idx in range(first_layer, len(layers)):
            modules.append(layers[layer_idx])
            names.append(f"encoder.layer.{layer_idx}")
        modules.append(encoder.layernorm)
        names.append("layernorm")

    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
    return names


def trainable_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def grad_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum().cpu())
    return total**0.5


def count_grad_tensors(parameters) -> int:
    return sum(1 for parameter in parameters if parameter.grad is not None)


def pairwise_distance_loss(reference: torch.Tensor, adapted: torch.Tensor) -> torch.Tensor:
    reference_dist = torch.cdist(reference, reference)
    adapted_dist = torch.cdist(adapted, adapted)
    denom = reference_dist.pow(2).mean().clamp_min(1e-12)
    return (adapted_dist - reference_dist).pow(2).mean() / denom


def pair_distance_error(reference: torch.Tensor, adapted: torch.Tensor) -> float:
    return float(pairwise_distance_loss(reference, adapted).detach().cpu())


def make_swd_directions(
    latent_dim: int, num_projections: int, seed: int, device: torch.device
) -> torch.Tensor:
    if num_projections < 1:
        raise ValueError("swd_projections must be at least 1")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    directions = torch.randn(num_projections, latent_dim, generator=generator)
    return F.normalize(directions, dim=-1).to(device)


def sliced_wasserstein_loss(
    reference: torch.Tensor,
    adapted: torch.Tensor,
    directions: torch.Tensor,
) -> torch.Tensor:
    if reference.shape != adapted.shape:
        raise ValueError(
            "Sliced Wasserstein inputs must have matching shapes; "
            f"got {tuple(reference.shape)} and {tuple(adapted.shape)}."
        )
    reference_projection = reference @ directions.T
    adapted_projection = adapted @ directions.T
    reference_sorted = reference_projection.sort(dim=0).values
    adapted_sorted = adapted_projection.sort(dim=0).values
    scale = reference_projection.var(dim=0, unbiased=False).mean().clamp_min(1e-6)
    return (adapted_sorted - reference_sorted).pow(2).mean() / scale


def normalized_joint_distribution(
    reference_blocks: tuple[torch.Tensor, ...],
    adapted_blocks: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(reference_blocks) != len(adapted_blocks):
        raise ValueError("Joint-distribution block counts must match")

    normalized_reference = []
    normalized_adapted = []
    for reference, adapted in zip(reference_blocks, adapted_blocks):
        if reference.shape != adapted.shape:
            raise ValueError("Joint-distribution block shapes must match")
        center = reference.mean(dim=0, keepdim=True)
        scale = (reference - center).pow(2).mean().sqrt().clamp_min(1e-4)
        block_weight = float(reference.shape[-1]) ** -0.5
        normalized_reference.append((reference - center) / scale * block_weight)
        normalized_adapted.append((adapted - center) / scale * block_weight)

    return (
        torch.cat(normalized_reference, dim=-1),
        torch.cat(normalized_adapted, dim=-1),
    )


def compute_losses(model, batch, args) -> tuple[torch.Tensor, dict]:
    (
        source_start_img,
        source_next_img,
        medium_start_img,
        medium_next_img,
        source_start,
        source_next,
        action_block,
    ) = batch

    aligned_start = model.encode({"pixels": medium_start_img.unsqueeze(1)})["emb"][:, 0]
    aligned_next = model.encode({"pixels": medium_next_img.unsqueeze(1)})["emb"][:, 0]
    pred_next = predict_next(model, aligned_start, action_block)
    source_pred_next = predict_next(model, source_start, action_block).detach()

    pair = 0.5 * (
        F.mse_loss(aligned_start, source_start)
        + F.mse_loss(aligned_next, source_next)
    )
    dyn_source = F.mse_loss(pred_next, source_next)
    target_dyn = F.mse_loss(pred_next, aligned_next)
    metric = 0.5 * (
        pairwise_distance_loss(source_start, aligned_start)
        + pairwise_distance_loss(source_next, aligned_next)
    )
    source_delta = source_pred_next - source_start
    target_delta = pred_next - aligned_start
    delta_metric = pairwise_distance_loss(source_delta, target_delta)
    latent_swd = sliced_wasserstein_loss(
        torch.cat([source_start, source_next], dim=0),
        torch.cat([aligned_start, aligned_next], dim=0),
        args.swd_directions,
    )
    transition_swd = sliced_wasserstein_loss(
        source_delta, target_delta, args.swd_directions
    )
    source_joint, target_joint = normalized_joint_distribution(
        (source_start, source_delta, action_block),
        (aligned_start, target_delta, action_block),
    )
    joint_swd = sliced_wasserstein_loss(
        source_joint, target_joint, args.joint_swd_directions
    )

    source_identity = 0.5 * (
        F.mse_loss(
            model.encode({"pixels": source_start_img.unsqueeze(1)})["emb"][:, 0],
            source_start,
        )
        + F.mse_loss(
            model.encode({"pixels": source_next_img.unsqueeze(1)})["emb"][:, 0],
            source_next,
        )
    )
    loss = (
        args.pair_weight * pair
        + args.dynamics_weight * dyn_source
        + args.target_dynamics_weight * target_dyn
        + args.metric_weight * metric
        + args.delta_metric_weight * delta_metric
        + args.latent_swd_weight * latent_swd
        + args.transition_swd_weight * transition_swd
        + args.joint_swd_weight * joint_swd
        + args.identity_weight * source_identity
    )
    metrics = {
        "loss": loss,
        "pair": pair,
        "dyn_source": dyn_source,
        "target_dyn": target_dyn,
        "metric": metric,
        "delta_metric": delta_metric,
        "latent_swd": latent_swd,
        "transition_swd": transition_swd,
        "joint_swd": joint_swd,
        "source_identity": source_identity,
    }
    return loss, metrics


def average_metrics(model, tensors, batch_size: int, device, args) -> dict:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for start in range(0, len(tensors[0]), batch_size):
            batch = [x[start : start + batch_size].to(device) for x in tensors]
            _loss, metrics = compute_losses(model, batch, args)
            batch_n = len(batch[0])
            count += batch_n
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * batch_n
    return {key: value / count for key, value in totals.items()}


def train_epoch(model, loader, optimizer, args, device) -> dict:
    model.eval()
    totals: dict[str, float] = {}
    count = 0

    for batch in loader:
        batch = [x.to(device) for x in batch]
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = compute_losses(model, batch, args)
        loss.backward()
        optimizer.step()

        batch_n = len(batch[0])
        count += batch_n
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * batch_n
    return {key: value / count for key, value in totals.items()}


def gradient_sanity_check(model, loader, args, device) -> dict:
    model.eval()
    model.zero_grad(set_to_none=True)
    batch = [x.to(device) for x in next(iter(loader))]
    medium_start = batch[2].detach().clone().requires_grad_(True)
    batch[2] = medium_start
    loss, _metrics = compute_losses(model, batch, args)
    loss.backward()

    trainable = trainable_parameters(model)
    frozen = [p for p in model.parameters() if not p.requires_grad]
    sanity = {
        "loss": float(loss.detach().cpu()),
        "encoder_grad_norm": grad_norm(trainable),
        "medium_start_grad_norm": (
            float(medium_start.grad.detach().norm().cpu())
            if medium_start.grad is not None
            else 0.0
        ),
        "encoder_grad_tensors": count_grad_tensors(trainable),
        "frozen_model_grad_tensors": count_grad_tensors(frozen),
    }
    model.zero_grad(set_to_none=True)
    return sanity


def encode_target_batches(model, images: torch.Tensor, batch_size: int, device):
    return encode_batches(model, images, batch_size, device)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cache_dir = resolve_cache_dir(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = swm.data.HDF5Dataset(
        args.dataset_name, keys_to_cache=["action"], cache_dir=cache_dir
    )
    total = args.train_samples + args.val_samples
    start_rows, next_rows, raw_action_blocks, start_data, next_data = (
        sample_transition_rows(dataset, total, args.seed, args.action_block)
    )

    print(f"Rendering {total} start/next paired source/medium transitions...")
    source_start_img, medium_start_img = render_medium_source(start_data)
    source_next_img, medium_next_img = render_medium_source(next_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = swm.wm.utils.load_pretrained(args.policy)
    model = model.to(device).eval()
    model.interpolate_pos_encoding = True

    print("Precomputing frozen source and raw medium latents...")
    model.requires_grad_(False)
    source_start = encode_batches(model, source_start_img, args.batch_size, device)
    source_next = encode_batches(model, source_next_img, args.batch_size, device)
    medium_start = encode_batches(model, medium_start_img, args.batch_size, device)
    medium_next = encode_batches(model, medium_next_img, args.batch_size, device)
    args.swd_directions = make_swd_directions(
        latent_dim=int(source_start.shape[-1]),
        num_projections=args.swd_projections,
        seed=args.seed + 17,
        device=device,
    )

    scaler = fit_action_scaler(dataset)
    action_blocks = standardize_action_blocks(
        raw_action_blocks,
        scaler,
        action_dim=raw_action_blocks.shape[-1] // args.action_block,
    )
    args.joint_swd_directions = make_swd_directions(
        latent_dim=2 * int(source_start.shape[-1]) + int(action_blocks.shape[-1]),
        num_projections=args.swd_projections,
        seed=args.seed + 29,
        device=device,
    )

    initial_checkpoint_report = None
    if args.init_encoder_checkpoint:
        initial_checkpoint = torch.load(
            args.init_encoder_checkpoint, map_location=device
        )
        initial_config = initial_checkpoint.get("adapter_config", {})
        if initial_config.get("type") != "encoder":
            raise ValueError(
                "init_encoder_checkpoint must contain an encoder adapter"
            )
        model.encoder.load_state_dict(
            initial_checkpoint["encoder_state_dict"], strict=True
        )
        initial_checkpoint_report = initial_checkpoint.get("report")
        print(f"Initialized encoder from {args.init_encoder_checkpoint}")

    trainable_module_names = set_trainable_encoder_depth(model, args.train_mode)
    trainable = trainable_parameters(model)
    trainable_count = sum(parameter.numel() for parameter in trainable)
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )

    tensors = (
        source_start_img,
        source_next_img,
        medium_start_img,
        medium_next_img,
        source_start,
        source_next,
        action_blocks,
    )
    train_slice = slice(0, args.train_samples)
    val_slice = slice(args.train_samples, total)
    train_tensors = tuple(x[train_slice] for x in tensors)
    val_tensors = tuple(x[val_slice] for x in tensors)
    loader = DataLoader(
        TensorDataset(*train_tensors),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

    baseline_pair = 0.5 * (
        F.mse_loss(medium_start[val_slice], source_start[val_slice])
        + F.mse_loss(medium_next[val_slice], source_next[val_slice])
    )
    with torch.no_grad():
        baseline_source_pred = predict_next(
            model,
            source_start[val_slice].to(device),
            action_blocks[val_slice].to(device),
        ).cpu()
        baseline_medium_pred = predict_next(
            model,
            medium_start[val_slice].to(device),
            action_blocks[val_slice].to(device),
        ).cpu()
    baseline_source_dyn = F.mse_loss(
        baseline_source_pred, source_next[val_slice]
    )
    baseline_medium_dyn = F.mse_loss(
        baseline_medium_pred, medium_next[val_slice]
    )
    baseline_dyn_source_target = F.mse_loss(
        baseline_medium_pred, source_next[val_slice]
    )
    baseline_distance = pair_distance_error(
        source_start[val_slice], medium_start[val_slice]
    )
    baseline_latent_swd = sliced_wasserstein_loss(
        torch.cat(
            [source_start[val_slice], source_next[val_slice]], dim=0
        ).to(device),
        torch.cat(
            [medium_start[val_slice], medium_next[val_slice]], dim=0
        ).to(device),
        args.swd_directions,
    )
    baseline_transition_swd = sliced_wasserstein_loss(
        baseline_source_pred - source_start[val_slice],
        baseline_medium_pred - medium_start[val_slice],
        args.swd_directions.cpu(),
    )

    initial_val = average_metrics(
        model, val_tensors, args.batch_size, device, args
    )

    grad_sanity = gradient_sanity_check(model, loader, args, device)
    print(
        "gradient_sanity "
        f"encoder_grad_norm={grad_sanity['encoder_grad_norm']:.6f} "
        f"medium_start_grad_norm={grad_sanity['medium_start_grad_norm']:.6f} "
        f"encoder_grad_tensors={grad_sanity['encoder_grad_tensors']} "
        f"frozen_model_grad_tensors={grad_sanity['frozen_model_grad_tensors']}"
    )
    print(
        f"train_mode={args.train_mode} trainable_params={trainable_count} "
        f"baseline_pair={float(baseline_pair):.6f} "
        f"baseline_source_dyn={float(baseline_source_dyn):.6f} "
        f"baseline_medium_dyn={float(baseline_medium_dyn):.6f}"
    )
    print(
        "initial_val "
        f"loss={initial_val['loss']:.6f} "
        f"pair={initial_val['pair']:.6f} "
        f"dyn={initial_val['dyn_source']:.6f} "
        f"swd={initial_val['latent_swd']:.6f} "
        f"dswd={initial_val['transition_swd']:.6f} "
        f"jswd={initial_val['joint_swd']:.6f}"
    )

    history = []
    best_val = float("inf")
    best_state = None
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        stats = train_epoch(model, loader, optimizer, args, device)
        val_stats = average_metrics(model, val_tensors, args.batch_size, device, args)
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in stats.items()},
            **{f"val_{key}": value for key, value in val_stats.items()},
        }
        history.append(row)

        objective = val_stats["loss"]
        if objective < best_val:
            best_val = objective
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.encoder.state_dict().items()
            }

        print(
            f"epoch {epoch:03d} "
            f"loss={stats['loss']:.6f} "
            f"pair={stats['pair']:.6f} "
            f"dyn={stats['dyn_source']:.6f} "
            f"tdyn={stats['target_dyn']:.6f} "
            f"metric={stats['metric']:.6f} "
            f"dmetric={stats['delta_metric']:.6f} "
            f"swd={stats['latent_swd']:.6f} "
            f"dswd={stats['transition_swd']:.6f} "
            f"jswd={stats['joint_swd']:.6f} "
            f"val_loss={val_stats['loss']:.6f} "
            f"val_pair={val_stats['pair']:.6f} "
            f"val_dyn={val_stats['dyn_source']:.6f} "
            f"val_target_dyn={val_stats['target_dyn']:.6f} "
            f"val_metric={val_stats['metric']:.6f} "
            f"val_dmetric={val_stats['delta_metric']:.6f} "
            f"val_swd={val_stats['latent_swd']:.6f} "
            f"val_dswd={val_stats['transition_swd']:.6f} "
            f"val_jswd={val_stats['joint_swd']:.6f}"
        )

    if best_state is not None:
        model.encoder.load_state_dict(best_state, strict=True)

    final_val = average_metrics(model, val_tensors, args.batch_size, device, args)
    adapted_val = encode_target_batches(
        model, medium_start_img[val_slice], args.batch_size, device
    )
    source_identity_val = encode_target_batches(
        model, source_start_img[val_slice], args.batch_size, device
    )
    final_pair_distance = pair_distance_error(
        source_start[val_slice], adapted_val
    )
    source_identity_mse = F.mse_loss(
        source_identity_val, source_start[val_slice]
    )

    adapter_config = {
        "type": "encoder",
        "train_mode": args.train_mode,
        "trainable_module_names": trainable_module_names,
        "distribution_alignment": {
            "method": "sliced_wasserstein",
            "projections": args.swd_projections,
        },
    }
    report = {
        "policy": args.policy,
        "dataset_name": args.dataset_name,
        "cache_dir": str(cache_dir),
        "seed": args.seed,
        "device": str(device),
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "action_block": args.action_block,
        "start_rows": [int(x) for x in start_rows],
        "next_rows": [int(x) for x in next_rows],
        "adapter_config": adapter_config,
        "trainable_parameters": trainable_count,
        "gradient_sanity": grad_sanity,
        "loss_config": {
            "pair_weight": args.pair_weight,
            "dynamics_weight": args.dynamics_weight,
            "target_dynamics_weight": args.target_dynamics_weight,
            "metric_weight": args.metric_weight,
            "delta_metric_weight": args.delta_metric_weight,
            "latent_swd_weight": args.latent_swd_weight,
            "transition_swd_weight": args.transition_swd_weight,
            "joint_swd_weight": args.joint_swd_weight,
            "swd_projections": args.swd_projections,
            "identity_weight": args.identity_weight,
        },
        "baseline_val_pair_mse": float(baseline_pair),
        "baseline_val_source_dynamics_mse": float(baseline_source_dyn),
        "baseline_val_medium_target_dynamics_mse": float(baseline_medium_dyn),
        "baseline_val_medium_to_source_dynamics_mse": float(
            baseline_dyn_source_target
        ),
        "baseline_val_pair_distance_error": baseline_distance,
        "baseline_val_latent_swd": float(baseline_latent_swd.cpu()),
        "baseline_val_transition_swd": float(baseline_transition_swd.cpu()),
        "init_encoder_checkpoint": args.init_encoder_checkpoint,
        "initial_checkpoint_report": initial_checkpoint_report,
        "initial_val_metrics": initial_val,
        "best_val_objective": best_val,
        "final_val_metrics": final_val,
        "final_val_pair_distance_error": final_pair_distance,
        "final_val_source_identity_mse": float(source_identity_mse),
        "history": history,
        "wall_clock_seconds": time.time() - start_time,
    }

    checkpoint = {
        "encoder_state_dict": model.encoder.state_dict(),
        "adapter_config": adapter_config,
        "report": report,
    }
    ckpt_path = output_dir / "encoder_latest.ckpt"
    torch.save(checkpoint, ckpt_path)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"best_val_objective: {best_val:.6f}")
    print(
        "final_val "
        f"pair={final_val['pair']:.6f} "
        f"dyn={final_val['dyn_source']:.6f} "
        f"target_dyn={final_val['target_dyn']:.6f} "
        f"metric={final_val['metric']:.6f} "
        f"delta_metric={final_val['delta_metric']:.6f} "
        f"latent_swd={final_val['latent_swd']:.6f} "
        f"transition_swd={final_val['transition_swd']:.6f} "
        f"joint_swd={final_val['joint_swd']:.6f} "
        f"distance={final_pair_distance:.6f}"
    )
    print(f"wrote: {ckpt_path.resolve()}")
    print(f"wrote: {(output_dir / 'report.json').resolve()}")


if __name__ == "__main__":
    main()
