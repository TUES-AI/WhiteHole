"""Train a MoVie-style spatial adaptive encoder on Reacher transitions."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reacher_distracting_control import (
    DCS_VARIANTS,
    DavisBearBackground,
    is_dcs_variant,
)
from scripts.reacher_movie_adapter import build_movie_encoder
from scripts.train_reacher_medium_encoder_adapter import (
    fit_action_scaler,
    make_swd_directions,
    normalize_images,
    normalized_joint_distribution,
    resolve_cache_dir,
    sample_transition_rows,
    sliced_wasserstein_loss,
    standardize_action_blocks,
)
from scripts.visualize_reacher_shifts import render_variant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train two shallow STNs and a low-rate visual encoder using only "
            "frozen-dynamics consistency on shifted Reacher transitions, with "
            "optional source-relative retention regularization."
        )
    )
    parser.add_argument("--dataset-name", default="dmc/reacher_random")
    parser.add_argument("--policy", default="quentinll/lewm-reacher")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--target-variant",
        choices=("medium_visual", "hard_camera", *DCS_VARIANTS),
        default="hard_camera",
    )
    parser.add_argument(
        "--output-dir",
        default="tmp_reacher_visualization/movie_adapter/hard_camera",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--val-samples", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--stn-hidden-channels", type=int, default=32)
    parser.add_argument("--stn-lr", type=float, default=1e-5)
    parser.add_argument("--encoder-lr", type=float, default=1e-7)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--freeze-projector", action="store_true")
    parser.add_argument("--source-identity-weight", type=float, default=0.0)
    parser.add_argument("--latent-swd-weight", type=float, default=0.0)
    parser.add_argument("--transition-swd-weight", type=float, default=0.0)
    parser.add_argument("--joint-swd-weight", type=float, default=0.0)
    parser.add_argument("--swd-projections", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--background-video", default=None)
    parser.add_argument("--background-seed", type=int, default=0)
    parser.add_argument("--episode-min", type=int, default=None)
    parser.add_argument("--episode-max", type=int, default=None)
    return parser.parse_args()


def render_images(
    rows: dict,
    variant: str,
    background_video: str | None = None,
    background_seed: int = 0,
) -> torch.Tensor:
    states = {"qpos": rows["qpos"], "qvel": rows["qvel"]}
    episode_key = "episode_idx" if "episode_idx" in rows else "ep_idx"
    frames = np.stack(
        render_variant(
            states,
            variant,
            image_size=224,
            background_video=background_video,
            background_seed=background_seed,
            background_episode_ids=rows[episode_key],
            background_step_indices=rows["step_idx"],
        )
    )
    return normalize_images(frames)


def predict_latent(model, start: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    action_embedding = model.action_encoder(actions.unsqueeze(1))
    return model.predict(start.unsqueeze(1), action_embedding)[:, -1]


def predict_next(
    model, start_images: torch.Tensor, next_images: torch.Tensor, actions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    start = model.encode({"pixels": start_images.unsqueeze(1)})["emb"][:, 0]
    target_next = model.encode({"pixels": next_images.unsqueeze(1)})["emb"][:, 0]
    predicted_next = predict_latent(model, start, actions)
    return start, target_next, predicted_next


def adaptation_loss(model, batch, args) -> tuple[torch.Tensor, dict]:
    (
        target_start_images,
        target_next_images,
        actions,
        source_start_images,
        source_next_images,
        source_start,
        source_next,
    ) = batch
    start, target_next, predicted_next = predict_next(
        model, target_start_images, target_next_images, actions
    )
    dynamics = F.mse_loss(predicted_next, target_next)

    source_predicted_next = predict_latent(model, source_start, actions).detach()
    source_delta = source_predicted_next - source_start
    target_delta = predicted_next - start
    latent_swd = sliced_wasserstein_loss(
        torch.cat([source_start, source_next], dim=0),
        torch.cat([start, target_next], dim=0),
        args.swd_directions,
    )
    transition_swd = sliced_wasserstein_loss(
        source_delta, target_delta, args.swd_directions
    )
    source_joint, target_joint = normalized_joint_distribution(
        (source_start, source_delta, actions),
        (start, target_delta, actions),
    )
    joint_swd = sliced_wasserstein_loss(
        source_joint, target_joint, args.joint_swd_directions
    )

    source_identity = dynamics.new_zeros(())
    if args.source_identity_weight > 0:
        current_source_start = model.encode(
            {"pixels": source_start_images.unsqueeze(1)}
        )["emb"][:, 0]
        current_source_next = model.encode(
            {"pixels": source_next_images.unsqueeze(1)}
        )["emb"][:, 0]
        source_identity = 0.5 * (
            F.mse_loss(current_source_start, source_start)
            + F.mse_loss(current_source_next, source_next)
        )

    loss = (
        dynamics
        + args.source_identity_weight * source_identity
        + args.latent_swd_weight * latent_swd
        + args.transition_swd_weight * transition_swd
        + args.joint_swd_weight * joint_swd
    )
    metrics = {
        "loss": float(loss.detach().cpu()),
        "dynamics_mse": float(dynamics.detach().cpu()),
        "source_identity": float(source_identity.detach().cpu()),
        "latent_swd": float(latent_swd.detach().cpu()),
        "transition_swd": float(transition_swd.detach().cpu()),
        "joint_swd": float(joint_swd.detach().cpu()),
        "start_std": float(start.std(dim=0, unbiased=False).mean().detach().cpu()),
        "next_std": float(
            target_next.std(dim=0, unbiased=False).mean().detach().cpu()
        ),
        "start_norm": float(start.norm(dim=-1).mean().detach().cpu()),
    }
    return loss, metrics


def average_metrics(model, tensors, batch_size: int, device: torch.device, args) -> dict:
    totals: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for start in range(0, len(tensors[0]), batch_size):
            batch = tuple(x[start : start + batch_size].to(device) for x in tensors)
            _loss, metrics = adaptation_loss(model, batch, args)
            n = len(batch[0])
            count += n
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * n
    return {key: value / count for key, value in totals.items()}


def encode_batches(model, images: torch.Tensor, batch_size: int, device: torch.device):
    outputs = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size].to(device)
            outputs.append(
                model.encode({"pixels": batch.unsqueeze(1)})["emb"][:, 0].cpu()
            )
    return torch.cat(outputs)


def grad_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum().cpu())
    return total**0.5


def state_rms_drift(module, initial_state: dict[str, torch.Tensor]) -> float:
    squared = 0.0
    count = 0
    for name, value in module.state_dict().items():
        if not value.is_floating_point():
            continue
        delta = value.detach().cpu() - initial_state[name]
        squared += float(delta.pow(2).sum())
        count += delta.numel()
    return math.sqrt(squared / max(1, count))


def affine_stats(movie_encoder, images: torch.Tensor, device: torch.device) -> dict:
    images = images.to(device)
    with torch.no_grad():
        input_theta = movie_encoder.input_stn.affine(images)
        warped = movie_encoder.input_stn(images)
        patch_grid = movie_encoder.base_encoder.embeddings.patch_embeddings.projection(
            warped
        )
        feature_theta = movie_encoder.feature_stn.affine(patch_grid)
    identity = input_theta.new_tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    return {
        "input_mean_theta": input_theta.mean(dim=0).cpu().tolist(),
        "input_mean_abs_identity_delta": float(
            (input_theta - identity).abs().mean().cpu()
        ),
        "feature_mean_theta": feature_theta.mean(dim=0).cpu().tolist(),
        "feature_mean_abs_identity_delta": float(
            (feature_theta - identity).abs().mean().cpu()
        ),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cache_dir = resolve_cache_dir(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = swm.data.load_dataset(
        args.dataset_name, keys_to_cache=["action"], cache_dir=str(cache_dir)
    )
    total = args.train_samples + args.val_samples
    start_rows, next_rows, raw_actions, start_data, next_data = sample_transition_rows(
        dataset,
        total,
        args.seed,
        args.action_block,
        episode_min=args.episode_min,
        episode_max=args.episode_max,
    )

    print(
        f"Rendering {total} {args.target_variant} start/next transitions "
        "for target-only MoVie supervision..."
    )
    target_start_images = render_images(
        start_data,
        args.target_variant,
        args.background_video,
        args.background_seed,
    )
    target_next_images = render_images(
        next_data,
        args.target_variant,
        args.background_video,
        args.background_seed,
    )
    source_start_images = render_images(start_data, "source")
    source_next_images = render_images(next_data, "source")

    scaler = fit_action_scaler(dataset)
    actions = standardize_action_blocks(
        raw_actions,
        scaler,
        action_dim=raw_actions.shape[-1] // args.action_block,
    )
    train_slice = slice(0, args.train_samples)
    val_slice = slice(args.train_samples, total)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = swm.wm.utils.load_pretrained(args.policy).to(device).eval()
    model.interpolate_pos_encoding = True
    model.encoder = build_movie_encoder(
        model.encoder,
        {"stn_hidden_channels": args.stn_hidden_channels},
    ).to(device)
    model.requires_grad_(False)

    source_start_latents = encode_batches(
        model, source_start_images, args.batch_size, device
    )
    source_next_latents = encode_batches(
        model, source_next_images, args.batch_size, device
    )
    latent_dim = int(source_start_latents.shape[-1])
    args.swd_directions = make_swd_directions(
        latent_dim,
        args.swd_projections,
        args.seed + 17,
        device,
    )
    args.joint_swd_directions = make_swd_directions(
        2 * latent_dim + int(actions.shape[-1]),
        args.swd_projections,
        args.seed + 29,
        device,
    )

    tensors = (
        target_start_images,
        target_next_images,
        actions,
        source_start_images,
        source_next_images,
        source_start_latents,
        source_next_latents,
    )
    train_tensors = tuple(tensor[train_slice] for tensor in tensors)
    val_tensors = tuple(tensor[val_slice] for tensor in tensors)
    loader = DataLoader(
        TensorDataset(*train_tensors),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

    stn_parameters = list(model.encoder.stn_parameters())
    visual_parameters = []
    if not args.freeze_encoder:
        visual_parameters.extend(model.encoder.encoder_parameters())
    if not args.freeze_projector:
        visual_parameters.extend(model.projector.parameters())
    for parameter in stn_parameters + visual_parameters:
        parameter.requires_grad = True
    if {id(p) for p in stn_parameters} & {id(p) for p in visual_parameters}:
        raise RuntimeError("STN and visual-encoder optimizer groups overlap")

    initial_encoder_state = {
        key: value.detach().cpu().clone()
        for key, value in model.encoder.base_encoder.state_dict().items()
    }
    initial_projector_state = {
        key: value.detach().cpu().clone()
        for key, value in model.projector.state_dict().items()
    }

    optimizer_groups = [{"params": stn_parameters, "lr": args.stn_lr}]
    if visual_parameters:
        optimizer_groups.append(
            {"params": visual_parameters, "lr": args.encoder_lr}
        )
    optimizer = torch.optim.Adam(optimizer_groups, weight_decay=args.weight_decay)

    baseline_val = average_metrics(
        model, val_tensors, args.batch_size, device, args
    )
    source_tensors = (
        source_start_images,
        source_next_images,
        actions,
        source_start_images,
        source_next_images,
        source_start_latents,
        source_next_latents,
    )
    source_val_tensors = tuple(tensor[val_slice] for tensor in source_tensors)
    baseline_source_val = average_metrics(
        model, source_val_tensors, args.batch_size, device, args
    )
    baseline_target_latents = encode_batches(
        model, target_start_images[val_slice], args.batch_size, device
    )
    baseline_source_latents = encode_batches(
        model, source_start_images[val_slice], args.batch_size, device
    )

    first_batch = tuple(x.to(device) for x in next(iter(loader)))
    model.zero_grad(set_to_none=True)
    sanity_loss, _sanity_metrics = adaptation_loss(model, first_batch, args)
    sanity_loss.backward()
    frozen_parameters = [p for p in model.parameters() if not p.requires_grad]
    gradient_sanity = {
        "loss": float(sanity_loss.detach().cpu()),
        "stn_grad_norm": grad_norm(stn_parameters),
        "visual_encoder_grad_norm": grad_norm(visual_parameters),
        "frozen_core_grad_tensors": sum(
            1 for parameter in frozen_parameters if parameter.grad is not None
        ),
        "optimizer_groups_overlap": False,
    }
    model.zero_grad(set_to_none=True)
    print(
        "gradient_sanity "
        f"stn={gradient_sanity['stn_grad_norm']:.6f} "
        f"visual={gradient_sanity['visual_encoder_grad_norm']:.6f} "
        f"frozen_core={gradient_sanity['frozen_core_grad_tensors']}"
    )
    print(
        f"baseline_val_dynamics={baseline_val['dynamics_mse']:.6f} "
        f"stn_params={sum(p.numel() for p in stn_parameters)} "
        f"visual_params={sum(p.numel() for p in visual_parameters)}"
    )

    best_val = float("inf")
    best_state = None
    history = []
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        totals: dict[str, float] = {}
        count = 0
        for batch in loader:
            batch = tuple(x.to(device) for x in batch)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = adaptation_loss(model, batch, args)
            loss.backward()
            optimizer.step()
            n = len(batch[0])
            count += n
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * n
        train_metrics = {key: value / count for key, value in totals.items()}
        val_metrics = average_metrics(
            model, val_tensors, args.batch_size, device, args
        )
        history.append(
            {
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
        )
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {
                "sae": {
                    key: value.detach().cpu().clone()
                    for key, value in model.encoder.state_dict().items()
                },
                "projector": {
                    key: value.detach().cpu().clone()
                    for key, value in model.projector.state_dict().items()
                },
            }
        if epoch == 1 or epoch == args.epochs or epoch % 8 == 0:
            print(
                f"epoch {epoch:03d} train_dyn={train_metrics['dynamics_mse']:.6f} "
                f"val_dyn={val_metrics['dynamics_mse']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} "
                f"val_swd={val_metrics['latent_swd']:.6f} "
                f"val_std={val_metrics['start_std']:.6f}"
            )

    if best_state is not None:
        model.encoder.load_state_dict(best_state["sae"], strict=True)
        model.projector.load_state_dict(best_state["projector"], strict=True)

    final_val = average_metrics(model, val_tensors, args.batch_size, device, args)
    final_source_val = average_metrics(
        model, source_val_tensors, args.batch_size, device, args
    )
    final_target_latents = encode_batches(
        model, target_start_images[val_slice], args.batch_size, device
    )
    final_source_latents = encode_batches(
        model, source_start_images[val_slice], args.batch_size, device
    )
    background_metadata = None
    if is_dcs_variant(args.target_variant):
        background_metadata = DavisBearBackground(
            args.background_video,
            dynamic=args.target_variant == "dcs_bear_dynamic",
            seed=args.background_seed,
        ).metadata()
    adapter_config = {
        "type": "movie_stn",
        "target_variant": args.target_variant,
        "stn_hidden_channels": args.stn_hidden_channels,
        "placements": ["rgb_input", "patch_projection_grid"],
        "encoder_adapted": not args.freeze_encoder,
        "projector_adapted": not args.freeze_projector,
    }
    report = {
        "method": "movie_style_with_optional_source_regularization",
        "policy": args.policy,
        "dataset_name": args.dataset_name,
        "cache_dir": str(cache_dir),
        "target_variant": args.target_variant,
        "background": background_metadata,
        "seed": args.seed,
        "device": str(device),
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "episode_range": [args.episode_min, args.episode_max],
        "start_rows": [int(x) for x in start_rows],
        "next_rows": [int(x) for x in next_rows],
        "adapter_config": adapter_config,
        "optimization": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer_steps": args.epochs
            * math.ceil(args.train_samples / args.batch_size),
            "stn_lr": args.stn_lr,
            "encoder_lr": args.encoder_lr,
            "weight_decay": args.weight_decay,
            "overlapping_optimizers": False,
        },
        "loss_config": {
            "dynamics_weight": 1.0,
            "source_identity_weight": args.source_identity_weight,
            "latent_swd_weight": args.latent_swd_weight,
            "transition_swd_weight": args.transition_swd_weight,
            "joint_swd_weight": args.joint_swd_weight,
            "swd_projections": args.swd_projections,
        },
        "trainable_parameters": {
            "stn": sum(p.numel() for p in stn_parameters),
            "visual_encoder": sum(p.numel() for p in visual_parameters),
        },
        "gradient_sanity": gradient_sanity,
        "baseline_val_metrics": baseline_val,
        "baseline_source_val_metrics": baseline_source_val,
        "best_val_objective": best_val,
        "final_val_metrics": final_val,
        "final_source_val_metrics": final_source_val,
        "paired_diagnostics_not_used_for_training": {
            "baseline_target_to_source_mse": float(
                F.mse_loss(baseline_target_latents, baseline_source_latents)
            ),
            "final_target_to_original_source_mse": float(
                F.mse_loss(final_target_latents, baseline_source_latents)
            ),
            "source_retention_mse": float(
                F.mse_loss(final_source_latents, baseline_source_latents)
            ),
        },
        "affine_diagnostics": affine_stats(
            model.encoder, target_start_images[val_slice], device
        ),
        "encoder_rms_drift": state_rms_drift(
            model.encoder.base_encoder, initial_encoder_state
        ),
        "projector_rms_drift": state_rms_drift(
            model.projector, initial_projector_state
        ),
        "history": history,
        "wall_clock_seconds": time.time() - start_time,
    }
    checkpoint = {
        "sae_state_dict": model.encoder.state_dict(),
        "projector_state_dict": model.projector.state_dict(),
        "adapter_config": adapter_config,
        "report": report,
    }
    checkpoint_path = output_dir / "movie_latest.ckpt"
    torch.save(checkpoint, checkpoint_path)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"best_val_objective: {best_val:.6f}")
    print(f"wrote: {checkpoint_path.resolve()}")
    print(f"wrote: {(output_dir / 'report.json').resolve()}")


if __name__ == "__main__":
    main()
