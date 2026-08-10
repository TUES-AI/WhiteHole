"""Train a small residual input CNN with frozen LeWM dynamics consistency."""

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

from scripts.reacher_conv_adapter import build_input_adapter
from scripts.reacher_distracting_control import (
    DCS_VARIANTS,
    DavisBearBackground,
    is_dcs_variant,
)
from scripts.train_reacher_medium_encoder_adapter import (
    fit_action_scaler,
    resolve_cache_dir,
    sample_transition_rows,
    standardize_action_blocks,
)
from scripts.train_reacher_movie_adapter import render_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an identity-initialized residual CNN before a frozen LeWM "
            "using target-domain one-step dynamics consistency."
        )
    )
    parser.add_argument("--dataset-name", default="dmc/reacher_random")
    parser.add_argument("--policy", default="quentinll/lewm-reacher")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--target-variant",
        choices=("medium_visual", "hard_camera", *DCS_VARIANTS),
        default="dcs_bear_dynamic",
    )
    parser.add_argument(
        "--output-dir",
        default="tmp_reacher_visualization/conv_dynamics_adapter",
    )
    parser.add_argument("--background-video", default=None)
    parser.add_argument("--background-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--val-samples", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument(
        "--architecture", choices=("conv", "coord_unet"), default="conv"
    )
    parser.add_argument("--channels", type=int, default=93)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--source-identity-weight", type=float, default=0.0)
    parser.add_argument("--source-pixel-weight", type=float, default=0.0)
    parser.add_argument("--target-source-latent-weight", type=float, default=0.0)
    parser.add_argument("--target-source-pixel-weight", type=float, default=0.0)
    parser.add_argument("--episode-min", type=int, default=0)
    parser.add_argument("--episode-max", type=int, default=127)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def device_for_run() -> torch.device:
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def encode_images(
    model,
    images: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    outputs = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size].to(device)
            outputs.append(
                model.encode({"pixels": batch.unsqueeze(1)})["emb"][:, 0].cpu()
            )
    return torch.cat(outputs)


def predict_latent(model, start: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    action_embedding = model.action_encoder(actions.unsqueeze(1))
    return model.predict(start.unsqueeze(1), action_embedding)[:, -1]


def adaptation_loss(model, adapter, batch, args) -> tuple[torch.Tensor, dict]:
    (
        target_start_images,
        target_next_images,
        actions,
        source_start_images,
        source_next_images,
        source_start_reference,
        source_next_reference,
    ) = batch

    adapted_start = adapter(target_start_images)
    adapted_next = adapter(target_next_images)
    start = model.encode({"pixels": adapted_start.unsqueeze(1)})["emb"][:, 0]
    target_next = model.encode({"pixels": adapted_next.unsqueeze(1)})["emb"][:, 0]
    predicted_next = predict_latent(model, start, actions)
    dynamics = F.mse_loss(predicted_next, target_next)

    target_source_latent = 0.5 * (
        F.mse_loss(start, source_start_reference)
        + F.mse_loss(target_next, source_next_reference)
    )
    target_source_pixel = 0.5 * (
        F.mse_loss(adapted_start, source_start_images)
        + F.mse_loss(adapted_next, source_next_images)
    )

    source_identity = dynamics.new_zeros(())
    source_pixel = dynamics.new_zeros(())
    if args.source_identity_weight > 0 or args.source_pixel_weight > 0:
        adapted_source_start = adapter(source_start_images)
        adapted_source_next = adapter(source_next_images)
        if args.source_identity_weight > 0:
            current_source_start = model.encode(
                {"pixels": adapted_source_start.unsqueeze(1)}
            )["emb"][:, 0]
            current_source_next = model.encode(
                {"pixels": adapted_source_next.unsqueeze(1)}
            )["emb"][:, 0]
            source_identity = 0.5 * (
                F.mse_loss(current_source_start, source_start_reference)
                + F.mse_loss(current_source_next, source_next_reference)
            )
        if args.source_pixel_weight > 0:
            source_pixel = 0.5 * (
                F.mse_loss(adapted_source_start, source_start_images)
                + F.mse_loss(adapted_source_next, source_next_images)
            )

    loss = (
        dynamics
        + args.source_identity_weight * source_identity
        + args.source_pixel_weight * source_pixel
        + args.target_source_latent_weight * target_source_latent
        + args.target_source_pixel_weight * target_source_pixel
    )
    return loss, {
        "loss": float(loss.detach().cpu()),
        "dynamics_mse": float(dynamics.detach().cpu()),
        "source_identity_mse": float(source_identity.detach().cpu()),
        "source_pixel_mse": float(source_pixel.detach().cpu()),
        "target_source_latent_mse": float(target_source_latent.detach().cpu()),
        "target_source_pixel_mse": float(target_source_pixel.detach().cpu()),
        "target_residual_abs_mean": float(
            (adapted_start - target_start_images).abs().mean().detach().cpu()
        ),
        "target_latent_std": float(
            start.std(dim=0, unbiased=False).mean().detach().cpu()
        ),
    }


def average_objective(
    model,
    adapter,
    tensors,
    batch_size: int,
    device: torch.device,
    args,
) -> dict:
    totals: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for start in range(0, len(tensors[0]), batch_size):
            batch = tuple(x[start : start + batch_size].to(device) for x in tensors)
            _, metrics = adaptation_loss(model, adapter, batch, args)
            n = len(batch[0])
            count += n
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * n
    return {key: value / count for key, value in totals.items()}


def domain_dynamics_metrics(
    model,
    adapter,
    start_images: torch.Tensor,
    next_images: torch.Tensor,
    actions: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict:
    total_mse = 0.0
    total_residual = 0.0
    count = 0
    with torch.no_grad():
        for offset in range(0, len(start_images), batch_size):
            start_image = start_images[offset : offset + batch_size].to(device)
            next_image = next_images[offset : offset + batch_size].to(device)
            action = actions[offset : offset + batch_size].to(device)
            adapted_start = adapter(start_image)
            adapted_next = adapter(next_image)
            start = model.encode({"pixels": adapted_start.unsqueeze(1)})["emb"][:, 0]
            target_next = model.encode(
                {"pixels": adapted_next.unsqueeze(1)}
            )["emb"][:, 0]
            predicted_next = predict_latent(model, start, action)
            n = len(start_image)
            count += n
            total_mse += float(F.mse_loss(predicted_next, target_next).cpu()) * n
            total_residual += float(
                (adapted_start - start_image).abs().mean().cpu()
            ) * n
    return {
        "dynamics_mse": total_mse / count,
        "residual_abs_mean": total_residual / count,
    }


def source_retention_metrics(
    model,
    adapter,
    start_images: torch.Tensor,
    next_images: torch.Tensor,
    start_reference: torch.Tensor,
    next_reference: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict:
    total_latent = 0.0
    total_residual = 0.0
    count = 0
    with torch.no_grad():
        for offset in range(0, len(start_images), batch_size):
            start_image = start_images[offset : offset + batch_size].to(device)
            next_image = next_images[offset : offset + batch_size].to(device)
            adapted_start = adapter(start_image)
            adapted_next = adapter(next_image)
            current_start = model.encode(
                {"pixels": adapted_start.unsqueeze(1)}
            )["emb"][:, 0]
            current_next = model.encode(
                {"pixels": adapted_next.unsqueeze(1)}
            )["emb"][:, 0]
            reference_start = start_reference[offset : offset + batch_size].to(device)
            reference_next = next_reference[offset : offset + batch_size].to(device)
            n = len(start_image)
            count += n
            total_latent += 0.5 * (
                float(F.mse_loss(current_start, reference_start).cpu())
                + float(F.mse_loss(current_next, reference_next).cpu())
            ) * n
            total_residual += 0.5 * (
                float((adapted_start - start_image).abs().mean().cpu())
                + float((adapted_next - next_image).abs().mean().cpu())
            ) * n
    return {
        "latent_identity_mse": total_latent / count,
        "residual_abs_mean": total_residual / count,
    }


def grad_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum().cpu())
    return total**0.5


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

    print(f"Rendering {total} {args.target_variant} start/next transitions...")
    target_start_images = render_images(
        start_data, args.target_variant, args.background_video, args.background_seed
    )
    target_next_images = render_images(
        next_data, args.target_variant, args.background_video, args.background_seed
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

    device = device_for_run()
    model = swm.wm.utils.load_pretrained(args.policy).to(device).eval()
    model.interpolate_pos_encoding = True
    model.requires_grad_(False)
    source_start_reference = encode_images(
        model, source_start_images, args.batch_size, device
    )
    source_next_reference = encode_images(
        model, source_next_images, args.batch_size, device
    )

    adapter = build_input_adapter(
        {
            "architecture": args.architecture,
            "channels": args.channels,
            "depth": args.depth,
            "base_channels": args.base_channels,
            "residual_scale": args.residual_scale,
        }
    ).to(device)
    adapter_parameters = list(adapter.parameters())
    optimizer = torch.optim.AdamW(
        adapter_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    tensors = (
        target_start_images,
        target_next_images,
        actions,
        source_start_images,
        source_next_images,
        source_start_reference,
        source_next_reference,
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

    baseline_target = domain_dynamics_metrics(
        model,
        adapter,
        target_start_images[val_slice],
        target_next_images[val_slice],
        actions[val_slice],
        args.batch_size,
        device,
    )
    baseline_source = domain_dynamics_metrics(
        model,
        adapter,
        source_start_images[val_slice],
        source_next_images[val_slice],
        actions[val_slice],
        args.batch_size,
        device,
    )
    baseline_source_retention = source_retention_metrics(
        model,
        adapter,
        source_start_images[val_slice],
        source_next_images[val_slice],
        source_start_reference[val_slice],
        source_next_reference[val_slice],
        args.batch_size,
        device,
    )

    first_batch = tuple(x.to(device) for x in next(iter(loader)))
    optimizer.zero_grad(set_to_none=True)
    sanity_loss, _ = adaptation_loss(model, adapter, first_batch, args)
    sanity_loss.backward()
    gradient_sanity = {
        "loss": float(sanity_loss.detach().cpu()),
        "adapter_grad_norm": grad_norm(adapter_parameters),
        "frozen_core_grad_tensors": sum(
            1 for parameter in model.parameters() if parameter.grad is not None
        ),
    }
    optimizer.zero_grad(set_to_none=True)
    print(
        f"baseline_target={baseline_target['dynamics_mse']:.6f} "
        f"baseline_source={baseline_source['dynamics_mse']:.6f} "
        f"adapter_params={sum(p.numel() for p in adapter_parameters)} "
        f"frozen_core_grads={gradient_sanity['frozen_core_grad_tensors']}"
    )

    best_val = float("inf")
    best_state = None
    history = []
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        totals: dict[str, float] = {}
        count = 0
        adapter.train()
        for batch in loader:
            batch = tuple(x.to(device) for x in batch)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = adaptation_loss(model, adapter, batch, args)
            loss.backward()
            optimizer.step()
            n = len(batch[0])
            count += n
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * n
        train_metrics = {key: value / count for key, value in totals.items()}
        adapter.eval()
        val_metrics = average_objective(
            model, adapter, val_tensors, args.batch_size, device, args
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
                key: value.detach().cpu().clone()
                for key, value in adapter.state_dict().items()
            }
        if epoch == 1 or epoch == args.epochs or epoch % 8 == 0:
            print(
                f"epoch {epoch:03d} "
                f"train_dyn={train_metrics['dynamics_mse']:.6f} "
                f"val_dyn={val_metrics['dynamics_mse']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} "
                f"residual={val_metrics['target_residual_abs_mean']:.6f}"
            )

    if best_state is not None:
        adapter.load_state_dict(best_state, strict=True)
    adapter.eval()
    final_target = domain_dynamics_metrics(
        model,
        adapter,
        target_start_images[val_slice],
        target_next_images[val_slice],
        actions[val_slice],
        args.batch_size,
        device,
    )
    final_source = domain_dynamics_metrics(
        model,
        adapter,
        source_start_images[val_slice],
        source_next_images[val_slice],
        actions[val_slice],
        args.batch_size,
        device,
    )
    final_source_retention = source_retention_metrics(
        model,
        adapter,
        source_start_images[val_slice],
        source_next_images[val_slice],
        source_start_reference[val_slice],
        source_next_reference[val_slice],
        args.batch_size,
        device,
    )

    background_metadata = None
    if is_dcs_variant(args.target_variant):
        background_metadata = DavisBearBackground(
            args.background_video,
            dynamic=args.target_variant == "dcs_bear_dynamic",
            seed=args.background_seed,
        ).metadata()
    adapter_config = {
        "type": "conv",
        "architecture": args.architecture,
        "channels": args.channels,
        "depth": args.depth,
        "base_channels": args.base_channels,
        "residual_scale": args.residual_scale,
        "target_variant": args.target_variant,
    }
    report = {
        "method": "residual_input_cnn_frozen_dynamics",
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
        "trainable_parameters": sum(p.numel() for p in adapter_parameters),
        "optimization": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer_steps": args.epochs
            * math.ceil(args.train_samples / args.batch_size),
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        "loss_config": {
            "dynamics_weight": 1.0,
            "source_identity_weight": args.source_identity_weight,
            "source_pixel_weight": args.source_pixel_weight,
            "target_source_latent_weight": args.target_source_latent_weight,
            "target_source_pixel_weight": args.target_source_pixel_weight,
        },
        "gradient_sanity": gradient_sanity,
        "baseline_target_val_metrics": baseline_target,
        "baseline_source_val_metrics": baseline_source,
        "baseline_source_retention": baseline_source_retention,
        "best_val_objective": best_val,
        "final_target_val_metrics": final_target,
        "final_source_val_metrics": final_source,
        "final_source_retention": final_source_retention,
        "history": history,
        "wall_clock_seconds": time.time() - start_time,
    }
    checkpoint = {
        "adapter_state_dict": adapter.state_dict(),
        "adapter_config": adapter_config,
        "report": report,
    }
    checkpoint_path = output_dir / "adapter_latest.ckpt"
    torch.save(checkpoint, checkpoint_path)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"best_val_objective: {best_val:.6f}")
    print(f"final_target_dynamics: {final_target['dynamics_mse']:.6f}")
    print(f"final_source_dynamics: {final_source['dynamics_mse']:.6f}")
    print(f"wrote: {checkpoint_path.resolve()}")


if __name__ == "__main__":
    main()
