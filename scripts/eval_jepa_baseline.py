import argparse
import dataclasses
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from pldm.data.dataset_factory import DatasetFactory
from pldm.data.utils import make_dataloader_for_prebatched_ds
from pldm.models.hjepa import HJEPA
from pldm.train import TrainConfig
from pldm_envs.wall.data.wall import WallDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lightweight diagnostics for a pretrained two-room JEPA baseline."
    )
    parser.add_argument("--config", default="configs/two_rooms_baseline_jepa.yaml")
    parser.add_argument(
        "--checkpoint",
        default=(
            "outputs/pldm/two_rooms_jepa_baseline_len17_3m/"
            "epoch=10_sample_step=2072576.ckpt"
        ),
    )
    parser.add_argument(
        "--output-json",
        default="outputs/eval/two_rooms_jepa_baseline_eval.json",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--probe-train-batches", type=int, default=48)
    parser.add_argument("--probe-val-batches", type=int, default=16)
    parser.add_argument("--probe-steps", type=int, default=400)
    parser.add_argument("--rollout-batches", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def take_batches(loader, n_batches, desc):
    batches = []
    iterator = iter(loader)
    for _ in tqdm(range(n_batches), desc=desc):
        try:
            batches.append(next(iterator))
        except StopIteration:
            iterator = iter(loader)
            batches.append(next(iterator))
    return batches


def build_model(config, sample, checkpoint_path):
    input_dim = sample.states.shape[2:]
    if len(input_dim) == 1:
        input_dim = input_dim[0]

    use_propio_pos = (
        hasattr(sample, "propio_pos")
        and sample.propio_pos is not None
        and bool(sample.propio_pos.shape[-1])
    )
    use_propio_vel = (
        hasattr(sample, "propio_vel")
        and sample.propio_vel is not None
        and bool(sample.propio_vel.shape[-1])
    )

    model = HJEPA(
        config.hjepa,
        input_dim=input_dim,
        normalizer=None,
        use_propio_pos=use_propio_pos,
        use_propio_vel=use_propio_vel,
    ).cuda()

    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    state_dict = {
        key.replace("_orig_mod.", ""): value
        for key, value in checkpoint["model_state_dict"].items()
    }
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


@torch.no_grad()
def encode_locations(model, batches):
    zs = []
    locs = []
    for batch in tqdm(batches, desc="Encoding batches"):
        states = batch.states.cuda().transpose(0, 1)
        enc = model.level1.forward_posterior(
            states,
            actions=None,
            encode_only=True,
        ).backbone_output.encodings

        locations = batch.locations.cuda().transpose(0, 1)
        t = min(enc.shape[0], locations.shape[0])
        zs.append(enc[:t].flatten(0, 1).detach())
        locs.append(locations[:t].flatten(0, 1).detach())
    return torch.cat(zs, dim=0), torch.cat(locs, dim=0)


def train_linear_probe(
    train_z,
    train_loc,
    val_z,
    val_loc,
    normalizer,
    steps,
):
    probe = torch.nn.Linear(train_z.shape[-1], train_loc.shape[-1]).cuda()
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    batch_size = min(4096, train_z.shape[0])

    for _ in tqdm(range(steps), desc="Training linear position probe"):
        idx = torch.randint(0, train_z.shape[0], (batch_size,), device=train_z.device)
        pred = probe(train_z[idx])
        loss = torch.nn.functional.mse_loss(pred, train_loc[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        pred = probe(val_z)
        mse_norm = torch.nn.functional.mse_loss(pred, val_loc).item()
        pred_px = normalizer.unnormalize_location(pred)
        loc_px = normalizer.unnormalize_location(val_loc)
        mse_px = torch.nn.functional.mse_loss(pred_px, loc_px).item()
        rmse_px = mse_px**0.5

        mean_loc = train_loc.mean(dim=0, keepdim=True).expand_as(val_loc)
        mean_px = normalizer.unnormalize_location(mean_loc)
        baseline_mse_px = torch.nn.functional.mse_loss(mean_px, loc_px).item()
        baseline_rmse_px = baseline_mse_px**0.5

    return {
        "linear_probe_mse_normalized": mse_norm,
        "linear_probe_rmse_pixels": rmse_px,
        "mean_location_baseline_rmse_pixels": baseline_rmse_px,
        "linear_probe_vs_mean_rmse_ratio": rmse_px / baseline_rmse_px,
    }


@torch.no_grad()
def rollout_metrics(model, batches):
    model.eval()
    pred_losses = []
    persistence_losses = []
    shuffled_action_losses = []

    for batch in tqdm(batches, desc="Evaluating latent rollouts"):
        states = batch.states.cuda().transpose(0, 1)
        actions = batch.actions.cuda().transpose(0, 1)

        forward = model.level1.forward_posterior(states, actions)
        encs = forward.backbone_output.encodings
        preds = forward.pred_output.predictions

        t = min(encs.shape[0], preds.shape[0])
        encs = encs[:t]
        preds = preds[:t]

        pred_losses.append(
            torch.nn.functional.mse_loss(preds[1:], encs[1:], reduction="none")
            .flatten(1)
            .mean(dim=1)
        )

        persistence = encs[0].unsqueeze(0).expand_as(encs[1:])
        persistence_losses.append(
            torch.nn.functional.mse_loss(persistence, encs[1:], reduction="none")
            .flatten(1)
            .mean(dim=1)
        )

        if actions.shape[1] > 1:
            shuffled = actions[:, torch.randperm(actions.shape[1], device=actions.device)]
            shuffled_forward = model.level1.forward_posterior(states, shuffled)
            shuffled_preds = shuffled_forward.pred_output.predictions[:t]
            shuffled_action_losses.append(
                torch.nn.functional.mse_loss(
                    shuffled_preds[1:], encs[1:], reduction="none"
                )
                .flatten(1)
                .mean(dim=1)
            )

    pred = torch.stack(pred_losses).mean(dim=0)
    persistence = torch.stack(persistence_losses).mean(dim=0)

    metrics = {
        "latent_rollout_mse_by_horizon": [x.item() for x in pred],
        "persistence_mse_by_horizon": [x.item() for x in persistence],
        "latent_rollout_vs_persistence_mse_ratio": (
            pred.mean() / persistence.mean()
        ).item(),
    }

    if shuffled_action_losses:
        shuffled = torch.stack(shuffled_action_losses).mean(dim=0)
        metrics["shuffled_action_mse_by_horizon"] = [x.item() for x in shuffled]
        metrics["latent_rollout_vs_shuffled_action_mse_ratio"] = (
            pred.mean() / shuffled.mean()
        ).item()

    return metrics


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    config = TrainConfig.parse_from_file(args.config)
    config.data.num_workers = 0
    config.data.offline_wall_config.batch_size = args.batch_size
    config.data.offline_wall_config.device = "cuda"
    config.data.wall_config.batch_size = args.batch_size
    config.data.wall_config.device = "cuda"
    config.quick_debug = False
    config.wandb = False

    datasets = DatasetFactory(
        config.data,
        probing_cfg=config.eval_cfg.probing,
        disable_l2=config.hjepa.disable_l2,
    ).create_datasets()
    train_loader = datasets.ds
    sample = next(iter(train_loader))
    normalizer = train_loader.normalizer

    model = build_model(config, sample, args.checkpoint)

    val_wall_config = dataclasses.replace(
        config.data.wall_config,
        train=False,
        size=args.batch_size * max(args.probe_val_batches, args.rollout_batches),
    )
    val_loader = make_dataloader_for_prebatched_ds(
        WallDataset(val_wall_config),
        loader_config=config.data,
        normalizer=normalizer,
    )

    train_batches = take_batches(
        train_loader, args.probe_train_batches, "Collecting probe train batches"
    )
    val_batches = take_batches(
        val_loader,
        max(args.probe_val_batches, args.rollout_batches),
        "Collecting eval batches",
    )

    train_z, train_loc = encode_locations(model, train_batches)
    val_z, val_loc = encode_locations(model, val_batches[: args.probe_val_batches])

    report = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "probe_train_samples": int(train_z.shape[0]),
        "probe_val_samples": int(val_z.shape[0]),
    }
    report.update(
        train_linear_probe(
            train_z=train_z,
            train_loc=train_loc,
            val_z=val_z,
            val_loc=val_loc,
            normalizer=normalizer,
            steps=args.probe_steps,
        )
    )
    report.update(rollout_metrics(model, val_batches[: args.rollout_batches]))

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
