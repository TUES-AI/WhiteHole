import argparse
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
    AdapterEvalConfig,
    AppearanceAdapter,
    AppearanceAdapterConfig,
    build_dataloaders,
    build_source_model,
    compute_losses,
    encode_locations,
    estimate_source_latent_scale,
    freeze_source_model,
    initialize_delta_from_pairs,
    jsonable,
    make_shifted_batch,
    resolve_device,
    rollout_metrics,
    seed_everything,
    set_source_model_for_adapter_train,
    take_batches,
)
from whitehole.configs import omegaconf_parse_files_vals


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose why a delta-vector appearance adapter can lower the "
            "dynamics objective while hurting source-compatible eval metrics."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/adaptation/two_rooms_medium_delta_proposal.yaml",
    )
    parser.add_argument(
        "--source-checkpoint",
        default=(
            "outputs/whitehole/two_rooms_jepa_baseline_len17_3m/"
            "epoch=10_sample_step=2072576.ckpt"
        ),
    )
    parser.add_argument(
        "--proposal-adapter-checkpoint",
        default=(
            "outputs/adaptation/two_rooms_medium_delta_proposal_3ep/"
            "adapter_latest.ckpt"
        ),
    )
    parser.add_argument(
        "--poc-adapter-checkpoint",
        default="outputs/adaptation/two_rooms_medium/adapter_latest.ckpt",
    )
    parser.add_argument("--data-path", default="outputs/data/two_rooms_len17_3m.npz")
    parser.add_argument("--appearance-shift", default="medium")
    parser.add_argument(
        "--output-json",
        default="outputs/eval/two_rooms_medium_adapter_diagnosis.json",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--probe-train-batches", type=int, default=32)
    parser.add_argument("--probe-val-batches", type=int, default=12)
    parser.add_argument("--probe-steps", type=int, default=250)
    parser.add_argument("--rollout-batches", type=int, default=12)
    parser.add_argument("--loss-batches", type=int, default=16)
    parser.add_argument("--pair-init-batches", type=int, default=64)
    parser.add_argument("--source-scale-batches", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def load_config(args) -> AdapterEvalConfig:
    config = omegaconf_parse_files_vals(AdapterEvalConfig, [args.config], [])
    config.source_checkpoint_path = args.source_checkpoint
    config.data.source_data_path = args.data_path
    config.data.appearance_shift = args.appearance_shift
    config.data.batch_size = args.batch_size
    config.data.num_workers = args.num_workers
    config.probe_train_batches = args.probe_train_batches
    config.probe_val_batches = args.probe_val_batches
    config.probe_steps = args.probe_steps
    config.rollout_batches = args.rollout_batches
    config.val_batches = max(args.loss_batches, args.probe_val_batches)
    config.source_scale_batches = args.source_scale_batches
    config.data.device = "cuda"
    return config


def make_adapter(latent_dim: int, device: torch.device) -> AppearanceAdapter:
    return AppearanceAdapter(AppearanceAdapterConfig(latent_dim=latent_dim)).to(device)


def load_checkpoint_adapter(
    path: str,
    latent_dim: int,
    device: torch.device,
) -> AppearanceAdapter:
    adapter = make_adapter(latent_dim, device)
    checkpoint = torch.load(path, map_location=device)
    adapter.load_state_dict(checkpoint["adapter_state_dict"], strict=True)
    adapter.eval()
    return adapter


@torch.no_grad()
def mean_encoded_locations(model, batches, device, adapter=None):
    return encode_locations(model, batches, device=device, adapter=adapter)


def fit_probe(train_z, train_loc, steps: int):
    probe = torch.nn.Linear(train_z.shape[-1], train_loc.shape[-1]).to(train_z.device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    batch_size = min(4096, train_z.shape[0])

    for _ in tqdm(range(steps), desc="Training diagnostic probe"):
        idx = torch.randint(0, train_z.shape[0], (batch_size,), device=train_z.device)
        pred = probe(train_z[idx])
        loss = F.mse_loss(pred, train_loc[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    probe.eval()
    return probe


@torch.no_grad()
def eval_probe(probe, z, loc, normalizer):
    pred = probe(z)
    mse_norm = F.mse_loss(pred, loc).item()
    pred_px = normalizer.unnormalize_location(pred)
    loc_px = normalizer.unnormalize_location(loc)
    rmse_px = F.mse_loss(pred_px, loc_px).item() ** 0.5
    return {
        "mse_normalized": mse_norm,
        "rmse_pixels": rmse_px,
    }


@torch.no_grad()
def encode_shifted_locations(model, batches, normalizer, shift, device, adapter=None):
    shifted = [make_shifted_batch(batch, normalizer, shift) for batch in batches]
    return mean_encoded_locations(model, shifted, device=device, adapter=adapter)


def average_objective_metrics(
    model,
    adapter,
    batches,
    normalizer,
    config,
    device,
    source_latent_scale,
):
    totals = {}
    adapter.eval()
    with torch.no_grad():
        for batch in tqdm(batches, desc="Evaluating objective losses"):
            _loss, metrics = compute_losses(
                model=model,
                adapter=adapter,
                source_batch=batch,
                normalizer=normalizer,
                appearance_shift=config.data.appearance_shift,
                objective_config=config.objectives,
                device=device,
                source_latent_scale=source_latent_scale,
            )
            for key, value in metrics.items():
                totals.setdefault(key, []).append(float(value.item()))

    return {key: sum(values) / len(values) for key, values in totals.items()}


@torch.no_grad()
def paired_mse_from_flat(source_z, target_z, adapter):
    adapted_z = adapter(target_z)
    return {
        "paired_mse_before": F.mse_loss(target_z, source_z).item(),
        "paired_mse_after": F.mse_loss(adapted_z, source_z).item(),
    }


def adapter_delta_report(adapter, pair_delta=None):
    delta = adapter.delta.detach()
    report = adapter.delta_stats()
    if pair_delta is not None:
        report["delta_to_pair_mean_l2"] = (delta - pair_delta).norm().item()
        pair_norm = pair_delta.norm().clamp_min(1e-12)
        delta_norm = delta.norm().clamp_min(1e-12)
        report["delta_pair_cosine"] = (
            torch.dot(delta, pair_delta).div(delta_norm * pair_norm).item()
        )
    return report


def main():
    args = parse_args()
    seed_everything(args.seed)
    torch.manual_seed(args.seed)

    config = load_config(args)
    device = resolve_device(config.data.device)

    baseline_config, train_loader, val_loader = build_dataloaders(
        source_config_path=config.source_config_path,
        data_config=config.data,
        val_batches=max(
            args.probe_val_batches, args.rollout_batches, args.loss_batches
        ),
    )
    sample = next(iter(train_loader))
    model = build_source_model(
        baseline_config=baseline_config,
        sample=sample,
        checkpoint_path=config.source_checkpoint_path,
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
        "Collecting diagnostic train batches",
    )
    val_batches = take_batches(
        val_loader,
        max(args.probe_val_batches, args.rollout_batches, args.loss_batches),
        "Collecting diagnostic val batches",
    )

    source_train_z, source_train_loc = mean_encoded_locations(
        model,
        train_batches,
        device=device,
    )
    source_val_z, source_val_loc = mean_encoded_locations(
        model,
        val_batches[: args.probe_val_batches],
        device=device,
    )
    target_train_z, target_train_loc = encode_shifted_locations(
        model,
        train_batches,
        train_loader.normalizer,
        config.data.appearance_shift,
        device=device,
    )
    target_val_z, target_val_loc = encode_shifted_locations(
        model,
        val_batches[: args.probe_val_batches],
        val_loader.normalizer,
        config.data.appearance_shift,
        device=device,
    )

    source_probe = fit_probe(source_train_z, source_train_loc, args.probe_steps)
    source_probe_report = eval_probe(
        source_probe,
        source_val_z,
        source_val_loc,
        train_loader.normalizer,
    )

    pair_adapter = make_adapter(latent_dim, device)
    initialize_delta_from_pairs(
        model=model,
        adapter=pair_adapter,
        loader=train_loader,
        normalizer=train_loader.normalizer,
        appearance_shift=config.data.appearance_shift,
        device=device,
        n_batches=args.pair_init_batches,
    )
    pair_delta = pair_adapter.delta.detach().clone()

    adapters = {
        "zero_delta": make_adapter(latent_dim, device),
        "pair_mean_delta": pair_adapter,
    }
    for name, path in (
        ("proposal_delta", args.proposal_adapter_checkpoint),
        ("previous_poc_delta", args.poc_adapter_checkpoint),
    ):
        if Path(path).exists():
            adapters[name] = load_checkpoint_adapter(path, latent_dim, device)
        else:
            print(f"Skipping missing adapter checkpoint: {path}")

    report = {
        "config": jsonable(config),
        "source_probe_on_source_val": source_probe_report,
        "source_latent_scale": source_latent_scale.item(),
        "probe_train_samples": int(source_train_z.shape[0]),
        "probe_val_samples": int(source_val_z.shape[0]),
        "adapters": {},
    }

    for name, adapter in adapters.items():
        adapter.eval()
        adapted_target_train_z = adapter(target_train_z)
        adapted_target_val_z = adapter(target_val_z)

        target_probe = fit_probe(
            adapted_target_train_z.detach(),
            target_train_loc,
            args.probe_steps,
        )

        adapter_report = {
            **adapter_delta_report(adapter, pair_delta=pair_delta),
            **paired_mse_from_flat(source_val_z, target_val_z, adapter),
            "proposal_objective": average_objective_metrics(
                model=model,
                adapter=adapter,
                batches=val_batches[: args.loss_batches],
                normalizer=val_loader.normalizer,
                config=config,
                device=device,
                source_latent_scale=source_latent_scale,
            ),
            "source_probe_on_adapted_target_val": eval_probe(
                source_probe,
                adapted_target_val_z,
                target_val_loc,
                train_loader.normalizer,
            ),
            "target_probe_on_adapted_target_val": eval_probe(
                target_probe,
                adapted_target_val_z,
                target_val_loc,
                train_loader.normalizer,
            ),
            "rollout": rollout_metrics(
                model=model,
                adapter=adapter,
                batches=val_batches[: args.rollout_batches],
                normalizer=val_loader.normalizer,
                appearance_shift=config.data.appearance_shift,
                device=device,
            ),
        }
        report["adapters"][name] = adapter_report

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(jsonable(report), indent=2) + "\n")
    print(json.dumps(jsonable(report), indent=2))


if __name__ == "__main__":
    main()
