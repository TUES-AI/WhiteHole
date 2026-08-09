import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from whitehole.adaptation.adapters import (
    AdapterDataConfig,
    AdapterFamily,
    AdapterObjectiveConfig,
    AppearanceAdapter,
    AppearanceAdapterConfig,
    build_dataloaders,
    build_source_model,
    compute_losses,
    encode_locations,
    estimate_source_latent_scale,
    freeze_source_model,
    jsonable,
    make_shifted_batch,
    resolve_device,
    seed_everything,
    set_source_model_for_adapter_train,
    take_batches,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose where appearance adapters lose source semantics."
    )
    parser.add_argument("--source-config", default="configs/two_rooms_baseline_jepa.yaml")
    parser.add_argument(
        "--source-checkpoint",
        default=(
            "outputs/whitehole/two_rooms_jepa_baseline_len17_3m/"
            "epoch=10_sample_step=2072576.ckpt"
        ),
    )
    parser.add_argument("--data-path", default="outputs/data/two_rooms_len17_3m.npz")
    parser.add_argument("--appearance-shift", default="medium")
    parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        help="Adapter as name:path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/eval/adapter_collapse/diagnosis.json",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/eval/adapter_collapse/diagnosis.csv",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--probe-train-batches", type=int, default=32)
    parser.add_argument("--probe-val-batches", type=int, default=12)
    parser.add_argument("--probe-steps", type=int, default=300)
    parser.add_argument("--loss-batches", type=int, default=8)
    parser.add_argument("--source-scale-batches", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def parse_adapter_specs(specs):
    parsed = []
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"Expected adapter spec name:path, got {spec!r}")
        name, path = spec.split(":", 1)
        parsed.append((name, path))
    return parsed


def adapter_config_from_checkpoint(checkpoint, latent_dim, device):
    raw = checkpoint.get("adapter_config", {})
    config = AppearanceAdapterConfig(latent_dim=latent_dim)
    for key, value in raw.items():
        if key == "family":
            value = AdapterFamily[str(value).split(".")[-1]]
        if hasattr(config, key):
            setattr(config, key, value)
    config.latent_dim = latent_dim
    return config


def load_adapter(path, latent_dim, device):
    checkpoint = torch.load(path, map_location=device)
    config = adapter_config_from_checkpoint(checkpoint, latent_dim, device)
    adapter = AppearanceAdapter(config).to(device)
    adapter.load_state_dict(checkpoint["adapter_state_dict"], strict=True)
    adapter.eval()
    return adapter


def make_identity_adapter(latent_dim, device):
    config = AppearanceAdapterConfig(
        family=AdapterFamily.ConstantOffset,
        latent_dim=latent_dim,
    )
    adapter = AppearanceAdapter(config).to(device)
    adapter.eval()
    return adapter


def fit_probe(train_z, train_loc, steps):
    probe = torch.nn.Linear(train_z.shape[-1], train_loc.shape[-1]).to(train_z.device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    batch_size = min(4096, train_z.shape[0])

    for _ in tqdm(range(steps), desc="Training collapse probe"):
        idx = torch.randint(0, train_z.shape[0], (batch_size,), device=train_z.device)
        loss = F.mse_loss(probe(train_z[idx]), train_loc[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    probe.eval()
    return probe


@torch.no_grad()
def eval_probe(probe, z, loc, normalizer):
    pred = probe(z)
    pred_px = normalizer.unnormalize_location(pred)
    loc_px = normalizer.unnormalize_location(loc)
    rmse_px = F.mse_loss(pred_px, loc_px).item() ** 0.5
    mean_loc = loc.mean(dim=0, keepdim=True).expand_as(loc)
    mean_px = normalizer.unnormalize_location(mean_loc)
    mean_rmse_px = F.mse_loss(mean_px, loc_px).item() ** 0.5
    return {
        "rmse_px": rmse_px,
        "rmse_vs_batch_mean_ratio": rmse_px / mean_rmse_px,
    }


@torch.no_grad()
def encode_shifted_locations(model, batches, normalizer, shift, device, adapter=None):
    shifted_batches = [
        make_shifted_batch(batch, normalizer, shift) for batch in batches
    ]
    return encode_locations(model, shifted_batches, device=device, adapter=adapter)


@torch.no_grad()
def latent_stats(z):
    centered = z - z.mean(dim=0, keepdim=True)
    variance = centered.pow(2).mean(dim=0)
    return {
        "latent_norm_mean": z.norm(dim=-1).mean().item(),
        "latent_total_variance": variance.sum().item(),
        "latent_mean_dim_variance": variance.mean().item(),
        "latent_active_dims_var_gt_1e-3": int((variance > 1e-3).sum().item()),
    }


@torch.no_grad()
def pair_stats(source_z, target_z, adapted_z):
    source_centered = source_z - source_z.mean(dim=0, keepdim=True)
    adapted_centered = adapted_z - adapted_z.mean(dim=0, keepdim=True)
    source_norm = source_centered.norm(dim=-1).clamp_min(1e-12)
    adapted_norm = adapted_centered.norm(dim=-1).clamp_min(1e-12)
    centered_cos = (
        source_centered.mul(adapted_centered).sum(dim=-1)
        / (source_norm * adapted_norm)
    )
    return {
        "pair_mse_before": F.mse_loss(target_z, source_z).item(),
        "pair_mse_after": F.mse_loss(adapted_z, source_z).item(),
        "pair_centered_cosine": centered_cos.mean().item(),
    }


@torch.no_grad()
def average_raw_objective(
    model,
    adapter,
    batches,
    normalizer,
    appearance_shift,
    device,
    source_latent_scale,
):
    objective = AdapterObjectiveConfig(
        alignment_weight=1.0,
        multistep_weight=1.0,
        local_isometry_weight=1.0,
        identity_prior_weight=1.0,
        pair_alignment_weight=1.0,
        source_identity_weight=1.0,
        variance_alignment_weight=1.0,
        covariance_alignment_weight=1.0,
        local_isometry_samples=32,
        covariance_samples=256,
    )
    totals = {}
    for batch in tqdm(batches, desc="Computing raw objective terms"):
        _loss, metrics = compute_losses(
            model=model,
            adapter=adapter,
            source_batch=batch,
            normalizer=normalizer,
            appearance_shift=appearance_shift,
            objective_config=objective,
            device=device,
            source_latent_scale=source_latent_scale,
        )
        for key, value in metrics.items():
            totals.setdefault(key, []).append(float(value.item()))

    return {key: sum(values) / len(values) for key, values in totals.items()}


def main():
    args = parse_args()
    seed_everything(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device("cuda")
    data_config = AdapterDataConfig(
        source_data_path=args.data_path,
        appearance_shift=args.appearance_shift,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device="cuda",
    )
    baseline_config, train_loader, val_loader = build_dataloaders(
        source_config_path=args.source_config,
        data_config=data_config,
        val_batches=max(args.probe_val_batches, args.loss_batches),
    )
    sample = next(iter(train_loader))
    model = build_source_model(
        baseline_config=baseline_config,
        sample=sample,
        checkpoint_path=args.source_checkpoint,
        device=device,
    )
    freeze_source_model(model, freeze_backbone=True, freeze_predictor=True)
    set_source_model_for_adapter_train(model)

    latent_dim = model.level1.repr_dim
    source_latent_scale = estimate_source_latent_scale(
        model=model,
        loader=train_loader,
        device=device,
        n_batches=args.source_scale_batches,
    )

    train_batches = take_batches(
        train_loader,
        args.probe_train_batches,
        "Collecting collapse train batches",
    )
    val_batches = take_batches(
        val_loader,
        max(args.probe_val_batches, args.loss_batches),
        "Collecting collapse val batches",
    )

    source_train_z, source_train_loc = encode_locations(
        model,
        train_batches,
        device=device,
    )
    source_val_z, source_val_loc = encode_locations(
        model,
        val_batches[: args.probe_val_batches],
        device=device,
    )
    target_train_z, target_train_loc = encode_shifted_locations(
        model,
        train_batches,
        train_loader.normalizer,
        args.appearance_shift,
        device=device,
    )
    target_val_z, target_val_loc = encode_shifted_locations(
        model,
        val_batches[: args.probe_val_batches],
        val_loader.normalizer,
        args.appearance_shift,
        device=device,
    )

    source_probe = fit_probe(source_train_z, source_train_loc, args.probe_steps)
    target_probe = fit_probe(target_train_z, target_train_loc, args.probe_steps)

    adapters = [("unadapted_target", None)]
    for name, path in parse_adapter_specs(args.adapter):
        adapters.append((name, load_adapter(path, latent_dim, device)))

    report = {
        "source_probe_on_source": eval_probe(
            source_probe,
            source_val_z,
            source_val_loc,
            train_loader.normalizer,
        ),
        "source_probe_on_unadapted_target": eval_probe(
            source_probe,
            target_val_z,
            target_val_loc,
            train_loader.normalizer,
        ),
        "target_self_probe_on_unadapted_target": eval_probe(
            target_probe,
            target_val_z,
            target_val_loc,
            train_loader.normalizer,
        ),
        "source_latent_stats": latent_stats(source_val_z),
        "target_latent_stats": latent_stats(target_val_z),
        "source_latent_scale": source_latent_scale.item(),
        "adapters": {},
    }

    rows = []
    for name, adapter in adapters:
        if adapter is None:
            adapter = make_identity_adapter(latent_dim, device)

        adapted_train_z = adapter(target_train_z).detach()
        adapted_val_z = adapter(target_val_z).detach()
        adapted_probe = fit_probe(
            adapted_train_z,
            target_train_loc,
            args.probe_steps,
        )

        source_transfer = eval_probe(
            source_probe,
            adapted_val_z,
            target_val_loc,
            train_loader.normalizer,
        )
        adapted_self = eval_probe(
            adapted_probe,
            adapted_val_z,
            target_val_loc,
            train_loader.normalizer,
        )
        row = {
            "name": name,
            "source_probe_rmse_px": source_transfer["rmse_px"],
            "source_probe_ratio": source_transfer["rmse_vs_batch_mean_ratio"],
            "self_probe_rmse_px": adapted_self["rmse_px"],
            "self_probe_ratio": adapted_self["rmse_vs_batch_mean_ratio"],
            **pair_stats(source_val_z, target_val_z, adapted_val_z),
            **latent_stats(adapted_val_z),
            **adapter.delta_stats(),
        }
        report["adapters"][name] = {
            **row,
            "raw_objective": average_raw_objective(
                model=model,
                adapter=adapter,
                batches=val_batches[: args.loss_batches],
                normalizer=val_loader.normalizer,
                appearance_shift=args.appearance_shift,
                device=device,
                source_latent_scale=source_latent_scale,
            ),
        }
        rows.append(row)

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(jsonable(report), indent=2) + "\n")

    fieldnames = sorted({key for row in rows for key in row})
    if "name" in fieldnames:
        fieldnames.remove("name")
        fieldnames = ["name", *fieldnames]

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(jsonable(report), indent=2))


if __name__ == "__main__":
    main()
