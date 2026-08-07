import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pldm.adaptation.adapters import (
    AdapterDataConfig,
    build_dataloaders,
    build_source_model,
    freeze_source_model,
    jsonable,
    resolve_device,
    seed_everything,
    take_batches,
)
from pldm_envs.wall.appearance import apply_appearance_shift


class InputChannelAffine(nn.Module):
    """Target-image channel calibration before the frozen encoder.

    This is intentionally tiny for the two-channel wall observations:

        x'_c = sum_j W[c, j] x_j + b[c]

    It is a FiLM-style input adapter/control, not a full encoder finetune.
    """

    def __init__(self, channels: int = 2):
        super().__init__()
        self.channels = channels
        self.matrix_delta = nn.Parameter(torch.zeros(channels, channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def matrix(self):
        eye = torch.eye(
            self.channels,
            device=self.matrix_delta.device,
            dtype=self.matrix_delta.dtype,
        )
        return eye + self.matrix_delta

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        weight = self.matrix()
        adapted = torch.einsum("ij,...jhw->...ihw", weight, states)
        bias = self.bias.view(*([1] * (states.ndim - 3)), self.channels, 1, 1)
        return adapted + bias

    @torch.no_grad()
    def stats(self):
        matrix = self.matrix().detach()
        return {
            "input_matrix": matrix.cpu().tolist(),
            "input_bias": self.bias.detach().cpu().tolist(),
            "input_matrix_delta_l2": self.matrix_delta.detach().norm().item(),
            "input_bias_l2": self.bias.detach().norm().item(),
            "adapter_trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a tiny input FiLM/channel-affine adapter before the frozen "
            "JEPA encoder with anti-collapse source-trajectory losses."
        )
    )
    parser.add_argument("--source-config", default="configs/two_rooms_baseline_jepa.yaml")
    parser.add_argument(
        "--source-checkpoint",
        default=(
            "outputs/pldm/two_rooms_jepa_baseline_len17_3m/"
            "epoch=10_sample_step=2072576.ckpt"
        ),
    )
    parser.add_argument("--data-path", default="outputs/data/two_rooms_len17_3m.npz")
    parser.add_argument("--appearance-shift", default="medium")
    parser.add_argument(
        "--output-dir",
        default="outputs/adaptation/input_film/two_rooms_medium_input_affine_3ep",
    )
    parser.add_argument(
        "--eval-json",
        default="outputs/eval/input_film/two_rooms_medium_input_affine_3ep_eval.json",
    )
    parser.add_argument(
        "--eval-csv",
        default="outputs/eval/input_film/two_rooms_medium_input_affine_3ep_eval.csv",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-batches-per-epoch", type=int, default=1000)
    parser.add_argument("--val-batches", type=int, default=32)
    parser.add_argument("--probe-train-batches", type=int, default=32)
    parser.add_argument("--probe-val-batches", type=int, default=12)
    parser.add_argument("--probe-steps", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--pair-weight", type=float, default=1.0)
    parser.add_argument("--source-rollout-weight", type=float, default=1.0)
    parser.add_argument("--self-rollout-weight", type=float, default=0.1)
    parser.add_argument("--variance-weight", type=float, default=1.0)
    parser.add_argument("--covariance-weight", type=float, default=0.05)
    parser.add_argument("--identity-weight", type=float, default=0.001)
    parser.add_argument("--image-pair-weight", type=float, default=0.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def encode_states_grad(model, states: torch.Tensor) -> torch.Tensor:
    return model.level1.backbone.forward_multiple(states).encodings


@torch.no_grad()
def encode_states(model, states: torch.Tensor) -> torch.Tensor:
    return encode_states_grad(model, states).detach()


def batch_to_device_time_major(batch, device):
    return (
        batch.states.to(device).transpose(0, 1),
        batch.actions.to(device).transpose(0, 1),
        batch.locations.to(device).transpose(0, 1),
    )


def flatten_time_batch(z):
    return z.reshape(-1, z.shape[-1])


def shift_states_on_device(states, normalizer, shift):
    if shift == "source":
        return states
    unnormalized_states = normalizer.unnormalize_state(states)
    shifted_states = apply_appearance_shift(unnormalized_states, shift)
    return normalizer.normalize_state(shifted_states)


def distribution_losses(adapted_z, source_z):
    adapted = flatten_time_batch(adapted_z)
    source = flatten_time_batch(source_z).detach()
    adapted_std = adapted.std(dim=0, unbiased=False)
    source_std = source.std(dim=0, unbiased=False)
    variance_loss = F.mse_loss(adapted_std, source_std)

    adapted_centered = adapted - adapted.mean(dim=0, keepdim=True)
    source_centered = source - source.mean(dim=0, keepdim=True)
    denom = max(1, adapted.shape[0] - 1)
    adapted_cov = adapted_centered.T @ adapted_centered / denom
    source_cov = source_centered.T @ source_centered / denom
    covariance_loss = F.mse_loss(adapted_cov, source_cov)
    return variance_loss, covariance_loss


def rollout_losses(model, adapted_z, source_z, actions, horizon, discount=1.0):
    steps = min(actions.shape[0], adapted_z.shape[0] - 1, source_z.shape[0] - 1, horizon)
    if steps <= 0:
        zero = adapted_z.new_zeros(())
        return zero, zero

    predictions = model.level1.forward_prior(
        adapted_z[0],
        repr_input=True,
        actions=actions[:steps],
        T=steps,
    ).pred_output.predictions

    source_losses = []
    self_losses = []
    weights = []
    for h in range(1, steps + 1):
        source_losses.append(F.mse_loss(predictions[h], source_z[h], reduction="none"))
        self_losses.append(F.mse_loss(predictions[h], adapted_z[h], reduction="none"))
        weights.append(discount ** (h - 1))

    source_by_h = torch.stack([loss.mean() for loss in source_losses])
    self_by_h = torch.stack([loss.mean() for loss in self_losses])
    weights_t = source_by_h.new_tensor(weights)
    weights_t = weights_t / weights_t.sum().clamp_min(1e-12)
    return (source_by_h * weights_t).sum(), (self_by_h * weights_t).sum()


def identity_loss(adapter):
    eye = torch.eye(
        adapter.channels,
        device=adapter.matrix_delta.device,
        dtype=adapter.matrix_delta.dtype,
    )
    return F.mse_loss(adapter.matrix(), eye) + adapter.bias.pow(2).mean()


def compute_losses(model, input_adapter, batch, normalizer, shift, args, device):
    source_states, actions, _locations = batch_to_device_time_major(batch, device)
    target_states = shift_states_on_device(source_states, normalizer, shift)

    with torch.no_grad():
        source_z = encode_states(model, source_states)
        target_z = encode_states(model, target_states)

    adapted_states = input_adapter(target_states)
    adapted_z = encode_states_grad(model, adapted_states)

    t = min(source_z.shape[0], target_z.shape[0], adapted_z.shape[0])
    source_z = source_z[:t]
    target_z = target_z[:t]
    adapted_z = adapted_z[:t]

    pair = F.mse_loss(adapted_z, source_z)
    source_rollout, self_rollout = rollout_losses(
        model=model,
        adapted_z=adapted_z,
        source_z=source_z,
        actions=actions[: max(0, t - 1)],
        horizon=args.horizon,
    )
    variance, covariance = distribution_losses(adapted_z, source_z)
    identity = identity_loss(input_adapter)

    if args.image_pair_weight:
        source_images = source_states[:t]
        image_pair = F.mse_loss(adapted_states[:t], source_images)
    else:
        image_pair = adapted_z.new_zeros(())

    total = (
        args.pair_weight * pair
        + args.source_rollout_weight * source_rollout
        + args.self_rollout_weight * self_rollout
        + args.variance_weight * variance
        + args.covariance_weight * covariance
        + args.identity_weight * identity
        + args.image_pair_weight * image_pair
    )

    with torch.no_grad():
        target_variance, target_covariance = distribution_losses(target_z, source_z)

    return total, {
        "loss": total.detach(),
        "pair_alignment_loss": pair.detach(),
        "source_rollout_loss": source_rollout.detach(),
        "self_rollout_loss": self_rollout.detach(),
        "variance_alignment_loss": variance.detach(),
        "covariance_alignment_loss": covariance.detach(),
        "identity_loss": identity.detach(),
        "image_pair_loss": image_pair.detach(),
        "unadapted_pair_alignment_loss": F.mse_loss(target_z, source_z).detach(),
        "unadapted_variance_alignment_loss": target_variance.detach(),
        "unadapted_covariance_alignment_loss": target_covariance.detach(),
    }


@torch.no_grad()
def latent_stats(z):
    flat = flatten_time_batch(z)
    centered = flat - flat.mean(dim=0, keepdim=True)
    variance = centered.pow(2).mean(dim=0)
    return {
        "latent_norm_mean": flat.norm(dim=-1).mean().item(),
        "latent_total_variance": variance.sum().item(),
        "latent_mean_dim_variance": variance.mean().item(),
        "latent_active_dims_var_gt_1e-3": int((variance > 1e-3).sum().item()),
    }


@torch.no_grad()
def encode_locations(model, batches, normalizer, shift, device, input_adapter=None):
    zs = []
    locs = []
    for batch in tqdm(batches, desc="Encoding eval locations"):
        states, _actions, locations = batch_to_device_time_major(batch, device)
        states = shift_states_on_device(states, normalizer, shift)
        if input_adapter is not None:
            states = input_adapter(states)
        z = encode_states(model, states)
        t = min(z.shape[0], locations.shape[0])
        zs.append(z[:t].reshape(-1, z.shape[-1]))
        locs.append(locations[:t].reshape(-1, locations.shape[-1]))
    return torch.cat(zs, dim=0), torch.cat(locs, dim=0)


def train_probe(train_z, train_loc, steps):
    probe = nn.Linear(train_z.shape[-1], train_loc.shape[-1]).to(train_z.device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    batch_size = min(4096, train_z.shape[0])

    for _ in tqdm(range(steps), desc="Training eval probe"):
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
    rmse = F.mse_loss(pred_px, loc_px).item() ** 0.5
    mean_loc = loc.mean(dim=0, keepdim=True).expand_as(loc)
    mean_px = normalizer.unnormalize_location(mean_loc)
    mean_rmse = F.mse_loss(mean_px, loc_px).item() ** 0.5
    return {
        "rmse_pixels": rmse,
        "rmse_vs_batch_mean_ratio": rmse / mean_rmse,
    }


@torch.no_grad()
def paired_and_rollout_eval(model, input_adapter, batches, normalizer, shift, device, horizon):
    before = []
    after = []
    source_rollouts = []
    self_rollouts = []
    persistence_losses = []
    adapted_latents = []
    target_latents = []
    source_latents = []

    for batch in tqdm(batches, desc="Evaluating input adapter"):
        source_states, actions, _locations = batch_to_device_time_major(batch, device)
        target_states = shift_states_on_device(source_states, normalizer, shift)

        source_z = encode_states(model, source_states)
        target_z = encode_states(model, target_states)
        adapted_z = encode_states(model, input_adapter(target_states))
        t = min(source_z.shape[0], target_z.shape[0], adapted_z.shape[0])
        source_z = source_z[:t]
        target_z = target_z[:t]
        adapted_z = adapted_z[:t]

        before.append(F.mse_loss(target_z, source_z).item())
        after.append(F.mse_loss(adapted_z, source_z).item())
        adapted_latents.append(adapted_z.detach())
        target_latents.append(target_z.detach())
        source_latents.append(source_z.detach())

        steps = min(actions.shape[0], adapted_z.shape[0] - 1, horizon)
        predictions = model.level1.forward_prior(
            adapted_z[0],
            repr_input=True,
            actions=actions[:steps],
            T=steps,
        ).pred_output.predictions
        source_rollouts.append(
            F.mse_loss(predictions[1:], source_z[1 : steps + 1], reduction="none")
            .flatten(1)
            .mean(dim=1)
        )
        self_rollouts.append(
            F.mse_loss(predictions[1:], adapted_z[1 : steps + 1], reduction="none")
            .flatten(1)
            .mean(dim=1)
        )
        persistence = adapted_z[0].unsqueeze(0).expand_as(adapted_z[1 : steps + 1])
        persistence_losses.append(
            F.mse_loss(persistence, adapted_z[1 : steps + 1], reduction="none")
            .flatten(1)
            .mean(dim=1)
        )

    adapted_all = torch.cat([flatten_time_batch(z) for z in adapted_latents], dim=0)
    target_all = torch.cat([flatten_time_batch(z) for z in target_latents], dim=0)
    source_all = torch.cat([flatten_time_batch(z) for z in source_latents], dim=0)
    source_rollout = torch.stack(source_rollouts).mean(dim=0)
    self_rollout = torch.stack(self_rollouts).mean(dim=0)
    persistence = torch.stack(persistence_losses).mean(dim=0)

    return {
        "paired_latent_mse_before_adapter": sum(before) / len(before),
        "paired_latent_mse_after_adapter": sum(after) / len(after),
        "adapted_latent_stats": latent_stats(adapted_all),
        "target_latent_stats": latent_stats(target_all),
        "source_latent_stats": latent_stats(source_all),
        "adapted_rollout_to_source_mse_by_horizon": [
            x.item() for x in source_rollout
        ],
        "adapted_self_rollout_mse_by_horizon": [x.item() for x in self_rollout],
        "adapted_persistence_mse_by_horizon": [x.item() for x in persistence],
        "adapted_self_rollout_vs_persistence_ratio": (
            self_rollout.mean() / persistence.mean().clamp_min(1e-12)
        ).item(),
        "adapted_source_rollout_mean": source_rollout.mean().item(),
    }


def evaluate(model, input_adapter, train_loader, val_loader, args, device):
    train_batches = take_batches(
        train_loader,
        args.probe_train_batches,
        "Collecting eval train batches",
    )
    val_batches = take_batches(
        val_loader,
        max(args.probe_val_batches, args.val_batches),
        "Collecting eval val batches",
    )
    source_train_z, source_train_loc = encode_locations(
        model,
        train_batches,
        train_loader.normalizer,
        "source",
        device,
    )
    source_val_z, source_val_loc = encode_locations(
        model,
        val_batches[: args.probe_val_batches],
        val_loader.normalizer,
        "source",
        device,
    )
    target_val_z, target_val_loc = encode_locations(
        model,
        val_batches[: args.probe_val_batches],
        val_loader.normalizer,
        args.appearance_shift,
        device,
    )
    adapted_train_z, adapted_train_loc = encode_locations(
        model,
        train_batches,
        train_loader.normalizer,
        args.appearance_shift,
        device,
        input_adapter=input_adapter,
    )
    adapted_val_z, adapted_val_loc = encode_locations(
        model,
        val_batches[: args.probe_val_batches],
        val_loader.normalizer,
        args.appearance_shift,
        device,
        input_adapter=input_adapter,
    )

    source_probe = train_probe(source_train_z, source_train_loc, args.probe_steps)
    adapted_probe = train_probe(adapted_train_z, adapted_train_loc, args.probe_steps)
    report = {
        "appearance_shift": args.appearance_shift,
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
        "source_probe_on_adapted_target": eval_probe(
            source_probe,
            adapted_val_z,
            adapted_val_loc,
            train_loader.normalizer,
        ),
        "adapted_self_probe_on_adapted_target": eval_probe(
            adapted_probe,
            adapted_val_z,
            adapted_val_loc,
            train_loader.normalizer,
        ),
    }
    report.update(
        paired_and_rollout_eval(
            model=model,
            input_adapter=input_adapter,
            batches=val_batches[: args.val_batches],
            normalizer=val_loader.normalizer,
            shift=args.appearance_shift,
            device=device,
            horizon=args.horizon,
        )
    )
    report.update(input_adapter.stats())
    return report


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    print(json.dumps({"event": "setup_start", "appearance_shift": args.appearance_shift}))

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
        val_batches=max(args.val_batches, args.probe_val_batches),
    )
    print(
        json.dumps(
            {
                "event": "dataloaders_ready",
                "train_batches": len(train_loader),
                "val_batches": len(val_loader),
            }
        )
    )
    sample = next(iter(train_loader))
    channels = sample.states.shape[-3]
    print(json.dumps({"event": "sample_ready", "channels": channels}))
    model = build_source_model(
        baseline_config=baseline_config,
        sample=sample,
        checkpoint_path=args.source_checkpoint,
        device=device,
    )
    print(json.dumps({"event": "source_model_ready"}))
    freeze_source_model(model, freeze_backbone=True, freeze_predictor=True)
    model.eval()
    model.level1.predictor.train()

    input_adapter = InputChannelAffine(channels=channels).to(device)
    optimizer = torch.optim.AdamW(
        input_adapter.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    history = []
    global_step = 0

    print(
        json.dumps(
            {
                "event": "input_film_train_start",
                "channels": channels,
                "train_batches": len(train_loader),
                "max_train_batches_per_epoch": args.max_train_batches_per_epoch,
                **input_adapter.stats(),
            },
            indent=2,
        )
    )

    for epoch in range(1, args.epochs + 1):
        totals = {}
        n_steps = 0
        for batch_idx, batch in (
            pbar := tqdm(enumerate(train_loader), total=len(train_loader), desc="Train")
        ):
            if batch_idx >= args.max_train_batches_per_epoch:
                break

            loss, metrics = compute_losses(
                model=model,
                input_adapter=input_adapter,
                batch=batch,
                normalizer=train_loader.normalizer,
                shift=args.appearance_shift,
                args=args,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    input_adapter.parameters(),
                    args.gradient_clip_norm,
                )
            optimizer.step()

            global_step += 1
            n_steps += 1
            for key, value in metrics.items():
                totals.setdefault(f"train/{key}", []).append(value.item())

            if global_step % 50 == 0:
                recent = {
                    key: sum(values[-50:]) / len(values[-50:])
                    for key, values in totals.items()
                }
                recent.update(input_adapter.stats())
                pbar.set_description(
                    f"loss={recent['train/loss']:.4f} "
                    f"pair={recent['train/pair_alignment_loss']:.4f} "
                    f"var={recent['train/variance_alignment_loss']:.4f}"
                )
                print(json.dumps({"step": global_step, **recent}))

        epoch_report = {
            "epoch": epoch,
            "step": global_step,
            **{key: sum(values) / len(values) for key, values in totals.items()},
            **input_adapter.stats(),
        }
        history.append(epoch_report)
        output_dir.joinpath("train_history.json").write_text(
            json.dumps(jsonable(history), indent=2) + "\n"
        )
        torch.save(
            {
                "adapter_state_dict": input_adapter.state_dict(),
                "epoch": epoch,
                "step": global_step,
                "args": vars(args),
                "metrics": jsonable(epoch_report),
            },
            output_dir / "adapter_latest.ckpt",
        )
        print(json.dumps(jsonable(epoch_report), indent=2))
        if n_steps == 0:
            raise RuntimeError("No training batches were processed.")

    eval_report = evaluate(
        model=model,
        input_adapter=input_adapter,
        train_loader=train_loader,
        val_loader=val_loader,
        args=args,
        device=device,
    )
    output_json = Path(args.eval_json)
    output_csv = Path(args.eval_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(jsonable(eval_report), indent=2) + "\n")

    flat = {
        "appearance_shift": eval_report["appearance_shift"],
        "source_probe_source_rmse": eval_report["source_probe_on_source"][
            "rmse_pixels"
        ],
        "source_probe_unadapted_target_rmse": eval_report[
            "source_probe_on_unadapted_target"
        ]["rmse_pixels"],
        "source_probe_adapted_target_rmse": eval_report[
            "source_probe_on_adapted_target"
        ]["rmse_pixels"],
        "adapted_self_probe_rmse": eval_report[
            "adapted_self_probe_on_adapted_target"
        ]["rmse_pixels"],
        "paired_latent_mse_before_adapter": eval_report[
            "paired_latent_mse_before_adapter"
        ],
        "paired_latent_mse_after_adapter": eval_report[
            "paired_latent_mse_after_adapter"
        ],
        "adapted_source_rollout_mean": eval_report["adapted_source_rollout_mean"],
        "adapted_total_variance": eval_report["adapted_latent_stats"][
            "latent_total_variance"
        ],
        "adapted_active_dims": eval_report["adapted_latent_stats"][
            "latent_active_dims_var_gt_1e-3"
        ],
        "target_total_variance": eval_report["target_latent_stats"][
            "latent_total_variance"
        ],
        "source_total_variance": eval_report["source_latent_stats"][
            "latent_total_variance"
        ],
        "input_matrix": json.dumps(eval_report["input_matrix"]),
        "input_bias": json.dumps(eval_report["input_bias"]),
    }
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)
    print(json.dumps(jsonable(eval_report), indent=2))


if __name__ == "__main__":
    main()
