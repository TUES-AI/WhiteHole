"""Train a small conv adapter for the Reacher medium visual shift."""

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
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reacher_conv_adapter import SmallConvAdapter
from scripts.visualize_reacher_shifts import render_variant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a residual CNN before LeWM for medium Reacher visuals."
    )
    parser.add_argument("--dataset-name", default="dmc/reacher_random")
    parser.add_argument("--policy", default="quentinll/lewm-reacher")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--output-dir",
        default="tmp_reacher_visualization/medium_conv_adapter",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--val-samples", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--pixel-weight", type=float, default=0.10)
    parser.add_argument("--identity-weight", type=float, default=0.02)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def resolve_cache_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    if os.environ.get("STABLEWM_HOME"):
        return Path(os.environ["STABLEWM_HOME"])
    return Path("stablewm_home")


def sample_rows(dataset, count: int, seed: int) -> tuple[np.ndarray, dict]:
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")

    # Avoid the first/last few states where resets and NaN actions are more common.
    valid = np.nonzero((step_idx >= 3) & (step_idx <= 190))[0]
    rng = np.random.default_rng(seed)
    row_indices = np.sort(rng.choice(valid, size=count, replace=False))
    return row_indices, dataset.get_row_data(row_indices)


def normalize_images(frames: np.ndarray) -> torch.Tensor:
    mean = torch.tensor(spt.data.dataset_stats.ImageNet["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(spt.data.dataset_stats.ImageNet["std"]).view(1, 3, 1, 1)
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    return (x - mean) / std


def render_pair_tensors(rows: dict) -> tuple[torch.Tensor, torch.Tensor]:
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
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size].to(device)
            out = model.encode({"pixels": batch.unsqueeze(1)})
            embs.append(out["emb"][:, 0].detach().cpu())
    return torch.cat(embs, dim=0)


def latent_mse(
    model,
    adapter,
    medium: torch.Tensor,
    source_emb: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> float:
    losses = []
    with torch.no_grad():
        for start in range(0, len(medium), batch_size):
            med = medium[start : start + batch_size].to(device)
            tgt = source_emb[start : start + batch_size].to(device)
            pred = model.encode({"pixels": adapter(med).unsqueeze(1)})["emb"][:, 0]
            losses.append(F.mse_loss(pred, tgt).detach().cpu())
    return float(torch.stack(losses).mean())


def train_epoch(
    model,
    adapter,
    loader,
    optimizer,
    args,
    device: torch.device,
) -> dict:
    adapter.train()
    totals = {"loss": 0.0, "latent": 0.0, "pixel": 0.0, "identity": 0.0}
    count = 0

    for source, medium, source_emb in loader:
        source = source.to(device)
        medium = medium.to(device)
        source_emb = source_emb.to(device)

        optimizer.zero_grad(set_to_none=True)
        adapted = adapter(medium)
        pred_emb = model.encode({"pixels": adapted.unsqueeze(1)})["emb"][:, 0]

        latent = F.mse_loss(pred_emb, source_emb)
        pixel = F.mse_loss(adapted, source)
        identity = F.mse_loss(adapter(source), source)
        loss = (
            latent
            + args.pixel_weight * pixel
            + args.identity_weight * identity
        )

        loss.backward()
        optimizer.step()

        batch_n = len(source)
        count += batch_n
        totals["loss"] += float(loss.detach().cpu()) * batch_n
        totals["latent"] += float(latent.detach().cpu()) * batch_n
        totals["pixel"] += float(pixel.detach().cpu()) * batch_n
        totals["identity"] += float(identity.detach().cpu()) * batch_n

    return {k: v / count for k, v in totals.items()}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cache_dir = resolve_cache_dir(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = swm.data.HDF5Dataset(args.dataset_name, cache_dir=cache_dir)
    total = args.train_samples + args.val_samples
    row_indices, rows = sample_rows(dataset, total, args.seed)

    print(f"Rendering {total} paired source/medium states...")
    source, medium = render_pair_tensors(rows)
    train_slice = slice(0, args.train_samples)
    val_slice = slice(args.train_samples, total)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = swm.wm.utils.load_pretrained(args.policy)
    model = model.to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    print("Precomputing frozen source embeddings...")
    source_emb = encode_batches(model, source, args.batch_size, device)
    medium_emb = encode_batches(model, medium, args.batch_size, device)
    baseline_train = float(
        F.mse_loss(medium_emb[train_slice], source_emb[train_slice])
    )
    baseline_val = float(F.mse_loss(medium_emb[val_slice], source_emb[val_slice]))

    adapter = SmallConvAdapter(
        channels=args.channels,
        depth=args.depth,
        residual_scale=args.residual_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    train_data = TensorDataset(
        source[train_slice], medium[train_slice], source_emb[train_slice]
    )
    loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

    history = []
    start_time = time.time()
    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        stats = train_epoch(model, adapter, loader, optimizer, args, device)
        adapter.eval()
        train_latent = latent_mse(
            model,
            adapter,
            medium[train_slice],
            source_emb[train_slice],
            args.batch_size,
            device,
        )
        val_latent = latent_mse(
            model,
            adapter,
            medium[val_slice],
            source_emb[val_slice],
            args.batch_size,
            device,
        )
        row = {
            "epoch": epoch,
            **stats,
            "train_latent_mse_after": train_latent,
            "val_latent_mse_after": val_latent,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} "
            f"loss={stats['loss']:.6f} "
            f"latent={stats['latent']:.6f} "
            f"val_latent={val_latent:.6f}"
        )
        if val_latent < best_val:
            best_val = val_latent
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in adapter.state_dict().items()
            }

    if best_state is not None:
        adapter.load_state_dict(best_state, strict=True)

    report = {
        "policy": args.policy,
        "dataset_name": args.dataset_name,
        "cache_dir": str(cache_dir),
        "seed": args.seed,
        "device": str(device),
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "row_indices": [int(x) for x in row_indices],
        "adapter_config": {
            "channels": args.channels,
            "depth": args.depth,
            "residual_scale": args.residual_scale,
        },
        "loss_config": {
            "pixel_weight": args.pixel_weight,
            "identity_weight": args.identity_weight,
        },
        "baseline_train_latent_mse": baseline_train,
        "baseline_val_latent_mse": baseline_val,
        "best_val_latent_mse": best_val,
        "history": history,
        "wall_clock_seconds": time.time() - start_time,
    }

    checkpoint = {
        "adapter_state_dict": adapter.state_dict(),
        "adapter_config": report["adapter_config"],
        "report": report,
    }
    ckpt_path = output_dir / "adapter_latest.ckpt"
    torch.save(checkpoint, ckpt_path)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"baseline_val_latent_mse: {baseline_val:.6f}")
    print(f"best_val_latent_mse: {best_val:.6f}")
    print(f"wrote: {ckpt_path.resolve()}")
    print(f"wrote: {(output_dir / 'report.json').resolve()}")


if __name__ == "__main__":
    main()
