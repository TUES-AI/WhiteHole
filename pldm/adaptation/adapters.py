import dataclasses
import json
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from pldm.configs import ConfigBase
from pldm.configs import omegaconf_parse_files_vals
from pldm_envs.wall.appearance import apply_appearance_shift


class AdapterFamily(Enum):
    ConstantOffset = auto()
    DiagonalAffine = auto()
    Orthogonal = auto()
    LowRank = auto()
    MLP = auto()


@dataclass
class AdapterDataConfig(ConfigBase):
    source_data_path: Optional[str] = "outputs/data/two_rooms_len17_3m.npz"
    target_data_path: Optional[str] = None
    appearance_shift: str = "medium"
    batch_size: int = 64
    n_steps: int = 16
    num_workers: int = 0
    device: str = "cuda"


@dataclass
class AdapterObjectiveConfig(ConfigBase):
    # Paper Eq. 10:
    #   P_phi(A(z_t), a_t) ~= A(z_{t+1})
    alignment_weight: float = 1.0

    # Paper Eq. 11:
    #   Sum_{h=2..H} w_h || zhat_{t+h} - A(z_{t+h}) ||^2
    multistep_weight: float = 1.0
    multistep_discount: float = 1.0

    # Paper Eq. 4 and Eq. 12. For ConstantOffset, Llocal is exactly zero.
    local_isometry_weight: float = 1.0
    identity_prior_weight: float = 1e-4
    horizon: int = 15
    source_scale_epsilon: float = 1e-6
    local_isometry_samples: int = 256

    # Synthetic two-room gives source/target pairs. Keep this diagnostic
    # available, but leave it off for proposal-style runs.
    pair_alignment_weight: float = 0.0
    source_identity_weight: float = 0.0
    variance_alignment_weight: float = 0.0
    covariance_alignment_weight: float = 0.0
    covariance_samples: int = 512


@dataclass
class AppearanceAdapterConfig(ConfigBase):
    family: AdapterFamily = AdapterFamily.ConstantOffset
    latent_dim: int = 512
    rank: int = 16
    hidden_dim: int = 512
    n_layers: int = 2
    zero_init: bool = True
    diagonal_scale_epsilon: float = 0.5
    residual_scale: float = 1.0
    use_layer_norm: bool = True


@dataclass
class AdapterTrainConfig(ConfigBase):
    source_config_path: str = "configs/two_rooms_baseline_jepa.yaml"
    source_checkpoint_path: str = (
        "outputs/pldm/two_rooms_jepa_baseline_len17_3m/"
        "epoch=10_sample_step=2072576.ckpt"
    )
    output_dir: str = "outputs/adaptation/two_rooms_medium_delta_proposal_3ep"
    adapter_checkpoint_path: str = (
        "outputs/adaptation/two_rooms_medium_delta_proposal_3ep/adapter_latest.ckpt"
    )
    output_json: str = (
        "outputs/eval/two_rooms_medium_delta_proposal_3ep_adapter_eval.json"
    )
    seed: int = 123
    epochs: int = 3
    lr: float = 1e-3
    weight_decay: float = 0.0
    delta_init_batches: int = 0
    source_scale_batches: int = 64
    max_train_batches_per_epoch: Optional[int] = 1500
    val_batches: int = 64
    log_every_n_steps: int = 50
    save_every_n_epochs: int = 1
    gradient_clip_norm: float = 1.0
    freeze_backbone: bool = True
    freeze_predictor: bool = True
    probe_train_batches: int = 48
    probe_val_batches: int = 16
    probe_steps: int = 400
    rollout_batches: int = 32
    data: AdapterDataConfig = field(default_factory=AdapterDataConfig)
    adapter: AppearanceAdapterConfig = field(default_factory=AppearanceAdapterConfig)
    objectives: AdapterObjectiveConfig = field(default_factory=AdapterObjectiveConfig)


@dataclass
class AdapterEvalConfig(AdapterTrainConfig):
    pass


class AppearanceAdapter(torch.nn.Module):
    """Target-latent -> source-latent appearance adapter.

    The delta-vector ablation uses:

        A(z_target) = z_target + delta

    The first higher-capacity adapter is a diagonal affine map:

        A(z_target) = (1 + epsilon * tanh(scale_logits)) * z_target + delta

    LowRank and MLP are residual maps:

        A(z_target) = z_target + delta + residual_scale * f(z_target)
    """

    def __init__(self, config: AppearanceAdapterConfig):
        super().__init__()
        self.config = config

        if config.family not in (
            AdapterFamily.ConstantOffset,
            AdapterFamily.DiagonalAffine,
            AdapterFamily.LowRank,
            AdapterFamily.MLP,
        ):
            raise NotImplementedError(
                "This adapter file currently implements ConstantOffset, "
                "DiagonalAffine, LowRank, and MLP."
            )

        self.delta = nn.Parameter(torch.zeros(config.latent_dim))
        if not config.zero_init:
            nn.init.normal_(self.delta, mean=0.0, std=0.02)

        if config.family == AdapterFamily.DiagonalAffine:
            self.scale_logits = nn.Parameter(torch.zeros(config.latent_dim))
            if not config.zero_init:
                nn.init.normal_(self.scale_logits, mean=0.0, std=0.02)

        if config.family == AdapterFamily.LowRank:
            self.low_rank_down = nn.Linear(config.latent_dim, config.rank, bias=False)
            self.low_rank_up = nn.Linear(config.rank, config.latent_dim, bias=False)
            if config.zero_init:
                nn.init.zeros_(self.low_rank_up.weight)

        if config.family == AdapterFamily.MLP:
            layers = []
            if config.use_layer_norm:
                layers.append(nn.LayerNorm(config.latent_dim))

            hidden_dim = int(config.hidden_dim)
            n_hidden = max(1, int(config.n_layers) - 1)
            in_dim = config.latent_dim
            for _ in range(n_hidden):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.SiLU())
                in_dim = hidden_dim

            final = nn.Linear(in_dim, config.latent_dim)
            if config.zero_init:
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)
            layers.append(final)
            self.residual_mlp = nn.Sequential(*layers)

    def diagonal_scale(self) -> torch.Tensor:
        epsilon = float(self.config.diagonal_scale_epsilon)
        return 1.0 + epsilon * self.scale_logits.tanh()

    def residual(self, z: torch.Tensor) -> torch.Tensor:
        if self.config.family == AdapterFamily.LowRank:
            return self.low_rank_up(self.low_rank_down(z))
        if self.config.family == AdapterFamily.MLP:
            return self.residual_mlp(z)
        return torch.zeros_like(z)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Map target-domain latents into source-compatible coordinates."""

        delta = self.delta.view(*([1] * (z.ndim - 1)), -1)
        if self.config.family == AdapterFamily.ConstantOffset:
            return z + delta

        if self.config.family == AdapterFamily.DiagonalAffine:
            scale = self.diagonal_scale().view(*([1] * (z.ndim - 1)), -1)
            return z * scale + delta

        return z + delta + float(self.config.residual_scale) * self.residual(z)

    @torch.no_grad()
    def set_delta(self, delta: torch.Tensor):
        if delta.shape != self.delta.shape:
            raise ValueError(
                f"Expected delta shape {self.delta.shape}, got {delta.shape}"
            )
        self.delta.copy_(delta)

    def delta_stats(self):
        with torch.no_grad():
            delta = self.delta.detach()
            stats = {
                "delta_l2": delta.norm().item(),
                "delta_mean_abs": delta.abs().mean().item(),
                "delta_max_abs": delta.abs().max().item(),
            }
            if hasattr(self, "scale_logits"):
                scale = self.diagonal_scale().detach()
                stats.update(
                    {
                        "scale_mean": scale.mean().item(),
                        "scale_std": scale.std(unbiased=False).item(),
                        "scale_min": scale.min().item(),
                        "scale_max": scale.max().item(),
                        "scale_logits_l2": self.scale_logits.detach().norm().item(),
                    }
                )
            residual_params = [
                param.detach().flatten()
                for name, param in self.named_parameters()
                if name != "delta"
            ]
            if residual_params:
                stats["residual_param_l2"] = torch.cat(residual_params).norm().item()
            stats["adapter_trainable_parameters"] = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
            return stats


def seed_everything(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)


def jsonable(value):
    if dataclasses.is_dataclass(value):
        return {k: jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, Enum):
        return f"{value.__class__.__name__}.{value.name}"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Adapter config requested cuda, but CUDA is unavailable.")
    return torch.device(device_name)


def build_baseline_train_config(
    source_config_path: str,
    data_config: AdapterDataConfig,
) -> object:
    from pldm.train import TrainConfig

    config = omegaconf_parse_files_vals(TrainConfig, [source_config_path], [])
    config.wandb = False
    config.quick_debug = False
    config.compile_model = False
    config.data.num_workers = data_config.num_workers

    if data_config.source_data_path is not None:
        config.data.offline_wall_config.offline_data_path = data_config.source_data_path

    config.data.offline_wall_config.batch_size = data_config.batch_size
    config.data.offline_wall_config.n_steps = data_config.n_steps
    config.data.offline_wall_config.device = data_config.device
    config.data.wall_config.batch_size = data_config.batch_size
    config.data.wall_config.n_steps = data_config.n_steps
    config.data.wall_config.device = data_config.device
    return config


def build_dataloaders(
    source_config_path: str,
    data_config: AdapterDataConfig,
    val_batches: int,
):
    from pldm.data.dataset_factory import DatasetFactory
    from pldm.data.utils import make_dataloader_for_prebatched_ds
    from pldm_envs.wall.data.wall import WallDataset

    if data_config.target_data_path is not None:
        raise NotImplementedError(
            "This first test expects a synthetic target made by applying "
            "appearance_shift to the source batch. Set target_data_path: null."
        )

    baseline_config = build_baseline_train_config(source_config_path, data_config)
    datasets = DatasetFactory(
        baseline_config.data,
        probing_cfg=baseline_config.eval_cfg.probing,
        disable_l2=baseline_config.hjepa.disable_l2,
    ).create_datasets()

    train_loader = datasets.ds
    val_wall_config = dataclasses.replace(
        baseline_config.data.wall_config,
        train=False,
        size=data_config.batch_size * max(1, val_batches),
    )
    val_loader = make_dataloader_for_prebatched_ds(
        WallDataset(val_wall_config),
        loader_config=baseline_config.data,
        normalizer=train_loader.normalizer,
    )

    return baseline_config, train_loader, val_loader


def infer_input_dim(sample):
    input_dim = sample.states.shape[2:]
    if len(input_dim) == 1:
        input_dim = input_dim[0]
    return input_dim


def uses_propio(sample, field: str) -> bool:
    return (
        hasattr(sample, field)
        and getattr(sample, field) is not None
        and bool(getattr(sample, field).shape[-1])
    )


def build_source_model(
    baseline_config,
    sample,
    checkpoint_path: str,
    device: torch.device,
):
    from pldm.models.hjepa import HJEPA

    model = HJEPA(
        baseline_config.hjepa,
        input_dim=infer_input_dim(sample),
        normalizer=None,
        use_propio_pos=uses_propio(sample, "propio_pos"),
        use_propio_vel=uses_propio(sample, "propio_vel"),
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = {
        key.replace("_orig_mod.", ""): value
        for key, value in checkpoint["model_state_dict"].items()
    }
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def freeze_source_model(
    model: torch.nn.Module, freeze_backbone: bool, freeze_predictor: bool
):
    if freeze_backbone:
        for param in model.level1.backbone.parameters():
            param.requires_grad = False
    if freeze_predictor:
        for param in model.level1.predictor.parameters():
            param.requires_grad = False


def set_source_model_for_adapter_train(model: torch.nn.Module):
    """Keep weights frozen while allowing cuDNN RNN backward through inputs."""

    model.eval()
    model.level1.predictor.train()


def make_shifted_batch(batch, normalizer, shift: str):
    if shift == "source":
        return batch

    states = normalizer.unnormalize_state(batch.states)
    shifted_states = apply_appearance_shift(states, shift)
    normalized_states = normalizer.normalize_state(shifted_states)
    return batch._replace(states=normalized_states)


def batch_to_device_time_major(batch, device: torch.device):
    states = batch.states.to(device).transpose(0, 1)
    actions = batch.actions.to(device).transpose(0, 1)
    locations = batch.locations.to(device).transpose(0, 1)
    return states, actions, locations


@torch.no_grad()
def encode_states(model: torch.nn.Module, states: torch.Tensor) -> torch.Tensor:
    return model.level1.forward_posterior(
        states,
        actions=None,
        encode_only=True,
    ).backbone_output.encodings.detach()


def dynamics_alignment_loss(
    adapter: nn.Module,
    jepa: nn.Module,
    z_t: torch.Tensor,
    actions: torch.Tensor,
    z_tp1: Optional[torch.Tensor] = None,
    horizon: Optional[int] = None,
) -> torch.Tensor:
    """Paper one-step intertwining loss.

    L_align = || P_phi(A(z_t), a_t) - A(z_{t+1}) ||^2
    """

    adapted_z = adapter(z_t)
    steps = min(actions.shape[0], adapted_z.shape[0] - 1)
    if horizon is not None and horizon > 0:
        steps = min(steps, horizon)
    if steps <= 0:
        return adapted_z.new_zeros(())

    level1 = jepa.level1 if hasattr(jepa, "level1") else jepa
    current = adapted_z[:steps].reshape(-1, adapted_z.shape[-1])
    if z_tp1 is None:
        target_next = adapted_z[1 : steps + 1].reshape(-1, adapted_z.shape[-1])
    else:
        target_next = adapter(z_tp1[:steps]).reshape(-1, adapted_z.shape[-1])
    one_step_actions = actions[:steps].reshape(-1, actions.shape[-1]).unsqueeze(0)

    predictions = level1.forward_prior(
        current,
        repr_input=True,
        actions=one_step_actions,
        T=1,
    ).pred_output.predictions[1]
    return F.mse_loss(predictions, target_next)


def multistep_rollout_loss(
    jepa: nn.Module,
    adapted_z: torch.Tensor,
    actions: torch.Tensor,
    horizon: int,
    discount: float = 1.0,
) -> torch.Tensor:
    """Paper multi-step rollout loss, starting at h=2."""

    steps = min(actions.shape[0], adapted_z.shape[0] - 1, horizon)
    if steps < 2:
        return adapted_z.new_zeros(())

    level1 = jepa.level1 if hasattr(jepa, "level1") else jepa
    predictions = level1.forward_prior(
        adapted_z[0],
        repr_input=True,
        actions=actions[:steps],
        T=steps,
    ).pred_output.predictions

    losses = []
    weights = []
    for h in range(2, steps + 1):
        losses.append(F.mse_loss(predictions[h], adapted_z[h], reduction="none"))
        weights.append(discount ** (h - 2))

    horizon_losses = torch.stack([loss.mean() for loss in losses])
    horizon_weights = horizon_losses.new_tensor(weights)
    horizon_weights = horizon_weights / horizon_weights.sum().clamp_min(1e-12)
    return (horizon_losses * horizon_weights).sum()


def local_isometry_loss(
    adapter: nn.Module,
    z: torch.Tensor,
    n_samples: int = 256,
) -> torch.Tensor:
    """Hutchinson estimate of ||J_A^T J_A - I||_F^2.

    Constant offsets preserve distances exactly, so their value is zero.
    """

    if isinstance(adapter, AppearanceAdapter):
        if adapter.config.family == AdapterFamily.ConstantOffset:
            return z.new_zeros(())
        if adapter.config.family == AdapterFamily.DiagonalAffine:
            scale = adapter.diagonal_scale()
            return (scale.pow(2) - 1.0).pow(2).sum()
        if adapter.config.family == AdapterFamily.LowRank:
            latent_dim = adapter.config.latent_dim
            weight = torch.eye(latent_dim, device=z.device, dtype=z.dtype)
            low_rank = adapter.low_rank_up.weight @ adapter.low_rank_down.weight
            weight = weight + float(adapter.config.residual_scale) * low_rank
            gram = weight.T @ weight
            identity = torch.eye(latent_dim, device=z.device, dtype=z.dtype)
            return (gram - identity).pow(2).sum()

    with torch.enable_grad():
        flat_z = z.detach().reshape(-1, z.shape[-1])
        if n_samples > 0 and flat_z.shape[0] > n_samples:
            idx = torch.randperm(flat_z.shape[0], device=flat_z.device)[:n_samples]
            flat_z = flat_z[idx]

        flat_z = flat_z.requires_grad_(True)
        v = torch.randn_like(flat_z)

        def apply_adapter(inp):
            return adapter(inp)

        _adapted, jvp = torch.autograd.functional.jvp(
            apply_adapter,
            flat_z,
            v,
            create_graph=True,
        )
        adapted = adapter(flat_z)
        vjp = torch.autograd.grad(
            adapted,
            flat_z,
            grad_outputs=jvp,
            create_graph=True,
            retain_graph=True,
        )[0]
        return (vjp - v).pow(2).sum(dim=-1).mean()


def identity_prior_loss(
    adapter: nn.Module,
    z: torch.Tensor,
    source_latent_scale: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    residual = adapter(z) - z
    residual_scale = residual.pow(2).sum(dim=-1).mean()
    return residual_scale / (source_latent_scale.to(z.device) + epsilon)


def source_identity_loss(adapter: nn.Module, source_z: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(adapter(source_z), source_z)


def distribution_alignment_losses(
    adapted_z: torch.Tensor,
    source_z: torch.Tensor,
    n_samples: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    adapted_flat = adapted_z.reshape(-1, adapted_z.shape[-1])
    source_flat = source_z.reshape(-1, source_z.shape[-1])
    n = min(adapted_flat.shape[0], source_flat.shape[0])
    adapted_flat = adapted_flat[:n]
    source_flat = source_flat[:n].detach()

    if n_samples > 0 and n > n_samples:
        idx = torch.randperm(n, device=adapted_flat.device)[:n_samples]
        adapted_flat = adapted_flat[idx]
        source_flat = source_flat[idx]

    adapted_std = adapted_flat.std(dim=0, unbiased=False)
    source_std = source_flat.std(dim=0, unbiased=False)
    variance_loss = F.mse_loss(adapted_std, source_std)

    if adapted_flat.shape[0] < 2:
        return variance_loss, adapted_flat.new_zeros(())

    adapted_centered = adapted_flat - adapted_flat.mean(dim=0, keepdim=True)
    source_centered = source_flat - source_flat.mean(dim=0, keepdim=True)
    denom = adapted_flat.shape[0] - 1
    adapted_cov = adapted_centered.T @ adapted_centered / denom
    source_cov = source_centered.T @ source_centered / denom
    covariance_loss = F.mse_loss(adapted_cov, source_cov)
    return variance_loss, covariance_loss


@torch.no_grad()
def estimate_source_latent_scale(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    n_batches: int,
) -> torch.Tensor:
    if n_batches <= 0:
        return torch.tensor(1.0, device=device)

    scales = []
    iterator = iter(loader)
    for _ in tqdm(range(n_batches), desc="Estimating source latent scale"):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        states, _actions, _locations = batch_to_device_time_major(batch, device)
        source_z = encode_states(model, states).reshape(-1, model.level1.repr_dim)
        scales.append(source_z.pow(2).sum(dim=-1).mean())

    return torch.stack(scales).mean().detach()


def compute_losses(
    model: torch.nn.Module,
    adapter: AppearanceAdapter,
    source_batch,
    normalizer,
    appearance_shift: str,
    objective_config: AdapterObjectiveConfig,
    device: torch.device,
    source_latent_scale: torch.Tensor,
):
    target_batch = make_shifted_batch(source_batch, normalizer, appearance_shift)
    source_states, actions, _locations = batch_to_device_time_major(
        source_batch, device
    )
    target_states, _target_actions, _target_locations = batch_to_device_time_major(
        target_batch,
        device,
    )

    source_z = encode_states(model, source_states)
    target_z = encode_states(model, target_states)
    adapted_z = adapter(target_z)
    t = min(source_z.shape[0], adapted_z.shape[0])

    alignment = dynamics_alignment_loss(
        adapter=adapter,
        jepa=model,
        z_t=target_z[:t],
        actions=actions[: max(0, t - 1)],
        horizon=objective_config.horizon,
    )
    multistep = multistep_rollout_loss(
        jepa=model,
        adapted_z=adapted_z[:t],
        actions=actions[: max(0, t - 1)],
        horizon=objective_config.horizon,
        discount=objective_config.multistep_discount,
    )
    zero_loss = adapted_z.new_zeros(())
    if objective_config.local_isometry_weight:
        local_iso = local_isometry_loss(
            adapter,
            target_z[:t],
            n_samples=objective_config.local_isometry_samples,
        )
    else:
        local_iso = zero_loss
    identity_prior = identity_prior_loss(
        adapter,
        target_z[:t],
        source_latent_scale=source_latent_scale,
        epsilon=objective_config.source_scale_epsilon,
    )
    pair_alignment = F.mse_loss(adapted_z[:t], source_z[:t])
    if objective_config.source_identity_weight:
        source_identity = source_identity_loss(adapter, source_z[:t])
    else:
        source_identity = zero_loss

    if (
        objective_config.variance_alignment_weight
        or objective_config.covariance_alignment_weight
    ):
        variance_alignment, covariance_alignment = distribution_alignment_losses(
            adapted_z[:t],
            source_z[:t],
            n_samples=objective_config.covariance_samples,
        )
    else:
        variance_alignment = zero_loss
        covariance_alignment = zero_loss

    total = (
        objective_config.alignment_weight * alignment
        + objective_config.multistep_weight * multistep
        + objective_config.local_isometry_weight * local_iso
        + objective_config.identity_prior_weight * identity_prior
        + objective_config.pair_alignment_weight * pair_alignment
        + objective_config.source_identity_weight * source_identity
        + objective_config.variance_alignment_weight * variance_alignment
        + objective_config.covariance_alignment_weight * covariance_alignment
    )

    with torch.no_grad():
        unadapted_pair_alignment = F.mse_loss(target_z[:t], source_z[:t])
        unadapted_dynamics_alignment = dynamics_alignment_loss(
            adapter=lambda z: z,
            jepa=model,
            z_t=target_z[:t],
            actions=actions[: max(0, t - 1)],
            horizon=objective_config.horizon,
        )

    metrics = {
        "loss": total,
        "alignment_loss": alignment.detach(),
        "multistep_loss": multistep.detach(),
        "local_isometry_loss": local_iso.detach(),
        "identity_prior_loss": identity_prior.detach(),
        "pair_alignment_loss": pair_alignment.detach(),
        "source_identity_loss": source_identity.detach(),
        "variance_alignment_loss": variance_alignment.detach(),
        "covariance_alignment_loss": covariance_alignment.detach(),
        "unadapted_pair_alignment_loss": unadapted_pair_alignment.detach(),
        "unadapted_dynamics_alignment_loss": unadapted_dynamics_alignment.detach(),
        "source_latent_scale": source_latent_scale.detach(),
    }
    return total, metrics


@torch.no_grad()
def initialize_delta_from_pairs(
    model: torch.nn.Module,
    adapter: AppearanceAdapter,
    loader,
    normalizer,
    appearance_shift: str,
    device: torch.device,
    n_batches: int,
):
    if n_batches <= 0:
        return {}

    total_delta = None
    total_count = 0
    iterator = iter(loader)
    for _ in tqdm(range(n_batches), desc="Initializing delta"):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        target_batch = make_shifted_batch(batch, normalizer, appearance_shift)
        source_states, _actions, _locations = batch_to_device_time_major(batch, device)
        target_states, _target_actions, _target_locations = batch_to_device_time_major(
            target_batch,
            device,
        )
        source_z = encode_states(model, source_states)
        target_z = encode_states(model, target_states)
        diff = (source_z - target_z).flatten(0, -2)

        batch_delta = diff.sum(dim=0)
        total_delta = batch_delta if total_delta is None else total_delta + batch_delta
        total_count += diff.shape[0]

    mean_delta = total_delta / max(1, total_count)
    adapter.set_delta(mean_delta)
    return {
        "delta_init_samples": total_count,
        "delta_init_l2": mean_delta.norm().item(),
        "delta_init_mean_abs": mean_delta.abs().mean().item(),
    }


@torch.no_grad()
def evaluate_loss_batches(
    model: torch.nn.Module,
    adapter: AppearanceAdapter,
    loader,
    normalizer,
    appearance_shift: str,
    objective_config: AdapterObjectiveConfig,
    device: torch.device,
    n_batches: int,
    source_latent_scale: torch.Tensor,
):
    adapter.eval()
    totals = {}
    iterator = iter(loader)

    for _ in tqdm(range(n_batches), desc="Validating adapter losses"):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        _loss, metrics = compute_losses(
            model=model,
            adapter=adapter,
            source_batch=batch,
            normalizer=normalizer,
            appearance_shift=appearance_shift,
            objective_config=objective_config,
            device=device,
            source_latent_scale=source_latent_scale,
        )
        for key, value in metrics.items():
            totals.setdefault(f"val/{key}", []).append(value.item())

    return {key: sum(values) / len(values) for key, values in totals.items()}


def save_adapter_checkpoint(
    output_dir: Path,
    name: str,
    adapter: AppearanceAdapter,
    optimizer: Optional[torch.optim.Optimizer],
    config: AdapterTrainConfig,
    epoch: int,
    step: int,
    metrics: dict,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "adapter_state_dict": adapter.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "train_config": jsonable(config),
        "adapter_config": jsonable(config.adapter),
        "epoch": epoch,
        "step": step,
        "metrics": jsonable(metrics),
    }
    torch.save(payload, output_dir / name)


def train_adapter(config: AdapterTrainConfig):
    seed_everything(config.seed)
    device = resolve_device(config.data.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(jsonable(config), indent=2) + "\n"
    )

    baseline_config, train_loader, val_loader = build_dataloaders(
        source_config_path=config.source_config_path,
        data_config=config.data,
        val_batches=config.val_batches,
    )
    sample = next(iter(train_loader))
    model = build_source_model(
        baseline_config=baseline_config,
        sample=sample,
        checkpoint_path=config.source_checkpoint_path,
        device=device,
    )
    freeze_source_model(model, config.freeze_backbone, config.freeze_predictor)
    set_source_model_for_adapter_train(model)

    latent_dim = model.level1.repr_dim
    if config.adapter.latent_dim != latent_dim:
        print(
            f"Adapter latent_dim={config.adapter.latent_dim} does not match "
            f"model repr_dim={latent_dim}; using {latent_dim}."
        )
        config.adapter.latent_dim = latent_dim

    adapter = AppearanceAdapter(config.adapter).to(device)
    source_latent_scale = estimate_source_latent_scale(
        model=model,
        loader=train_loader,
        device=device,
        n_batches=config.source_scale_batches,
    )
    init_metrics = initialize_delta_from_pairs(
        model=model,
        adapter=adapter,
        loader=train_loader,
        normalizer=train_loader.normalizer,
        appearance_shift=config.data.appearance_shift,
        device=device,
        n_batches=config.delta_init_batches,
    )

    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    global_step = 0
    best_val = float("inf")
    history = []

    print(
        json.dumps(
            {
                "event": "adapter_train_start",
                "appearance_shift": config.data.appearance_shift,
                "train_batches_per_epoch": len(train_loader),
                "max_train_batches_per_epoch": config.max_train_batches_per_epoch,
                "val_batches": config.val_batches,
                "source_latent_scale": source_latent_scale.item(),
                **init_metrics,
                **adapter.delta_stats(),
            },
            indent=2,
        )
    )

    for epoch in range(1, config.epochs + 1):
        adapter.train()
        set_source_model_for_adapter_train(model)
        epoch_metrics = {}
        n_steps = 0

        for batch_idx, batch in (
            pbar := tqdm(enumerate(train_loader), total=len(train_loader), desc="Train")
        ):
            if (
                config.max_train_batches_per_epoch is not None
                and batch_idx >= config.max_train_batches_per_epoch
            ):
                break

            loss, metrics = compute_losses(
                model=model,
                adapter=adapter,
                source_batch=batch,
                normalizer=train_loader.normalizer,
                appearance_shift=config.data.appearance_shift,
                objective_config=config.objectives,
                device=device,
                source_latent_scale=source_latent_scale,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    adapter.parameters(),
                    config.gradient_clip_norm,
                )
            optimizer.step()

            global_step += 1
            n_steps += 1
            for key, value in metrics.items():
                epoch_metrics.setdefault(f"train/{key}", []).append(value.item())

            if global_step % config.log_every_n_steps == 0:
                log = {
                    key: sum(values[-config.log_every_n_steps :])
                    / len(values[-config.log_every_n_steps :])
                    for key, values in epoch_metrics.items()
                }
                log.update(adapter.delta_stats())
                pbar.set_description(
                    f"loss={log['train/loss']:.5f} "
                    f"align={log['train/alignment_loss']:.5f}"
                )
                print(json.dumps({"step": global_step, **log}))

        train_epoch_metrics = {
            key: sum(values) / len(values) for key, values in epoch_metrics.items()
        }
        val_metrics = evaluate_loss_batches(
            model=model,
            adapter=adapter,
            loader=val_loader,
            normalizer=train_loader.normalizer,
            appearance_shift=config.data.appearance_shift,
            objective_config=config.objectives,
            device=device,
            n_batches=config.val_batches,
            source_latent_scale=source_latent_scale,
        )
        epoch_report = {
            "epoch": epoch,
            "step": global_step,
            **train_epoch_metrics,
            **val_metrics,
            **adapter.delta_stats(),
        }
        history.append(epoch_report)
        (output_dir / "train_history.json").write_text(
            json.dumps(jsonable(history), indent=2) + "\n"
        )
        print(json.dumps(epoch_report, indent=2))

        val_key = "val/loss"
        if val_metrics[val_key] < best_val:
            best_val = val_metrics[val_key]
            save_adapter_checkpoint(
                output_dir=output_dir,
                name="adapter_best.ckpt",
                adapter=adapter,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                step=global_step,
                metrics=epoch_report,
            )

        if epoch % config.save_every_n_epochs == 0 or epoch == config.epochs:
            save_adapter_checkpoint(
                output_dir=output_dir,
                name=f"adapter_epoch={epoch}_step={global_step}.ckpt",
                adapter=adapter,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                step=global_step,
                metrics=epoch_report,
            )
            save_adapter_checkpoint(
                output_dir=output_dir,
                name="adapter_latest.ckpt",
                adapter=adapter,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                step=global_step,
                metrics=epoch_report,
            )

        if n_steps == 0:
            raise RuntimeError("No training batches were processed.")

    return history[-1]


def take_batches(loader, n_batches: int, desc: str):
    batches = []
    iterator = iter(loader)
    for _ in tqdm(range(n_batches), desc=desc):
        try:
            batches.append(next(iterator))
        except StopIteration:
            iterator = iter(loader)
            batches.append(next(iterator))
    return batches


@torch.no_grad()
def encode_locations(
    model: torch.nn.Module, batches, device: torch.device, adapter=None
):
    zs = []
    locs = []
    for batch in tqdm(batches, desc="Encoding locations"):
        states, _actions, locations = batch_to_device_time_major(batch, device)
        z = encode_states(model, states)
        if adapter is not None:
            z = adapter(z)
        t = min(z.shape[0], locations.shape[0])
        zs.append(z[:t].reshape(-1, z.shape[-1]).detach())
        locs.append(locations[:t].reshape(-1, locations.shape[-1]).detach())
    return torch.cat(zs, dim=0), torch.cat(locs, dim=0)


def train_linear_probe(
    train_z,
    train_loc,
    val_z,
    val_loc,
    normalizer,
    steps: int,
):
    probe = nn.Linear(train_z.shape[-1], train_loc.shape[-1]).to(train_z.device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    batch_size = min(4096, train_z.shape[0])

    for _ in tqdm(range(steps), desc="Training eval probe"):
        idx = torch.randint(0, train_z.shape[0], (batch_size,), device=train_z.device)
        pred = probe(train_z[idx])
        loss = F.mse_loss(pred, train_loc[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        pred = probe(val_z)
        mse_norm = F.mse_loss(pred, val_loc).item()
        pred_px = normalizer.unnormalize_location(pred)
        loc_px = normalizer.unnormalize_location(val_loc)
        rmse_px = F.mse_loss(pred_px, loc_px).item() ** 0.5

        mean_loc = train_loc.mean(dim=0, keepdim=True).expand_as(val_loc)
        mean_px = normalizer.unnormalize_location(mean_loc)
        baseline_rmse_px = F.mse_loss(mean_px, loc_px).item() ** 0.5

    return {
        "linear_probe_mse_normalized": mse_norm,
        "linear_probe_rmse_pixels": rmse_px,
        "mean_location_baseline_rmse_pixels": baseline_rmse_px,
        "linear_probe_vs_mean_rmse_ratio": rmse_px / baseline_rmse_px,
    }


@torch.no_grad()
def paired_latent_metrics(
    model: torch.nn.Module,
    adapter: AppearanceAdapter,
    batches,
    normalizer,
    appearance_shift: str,
    device: torch.device,
):
    before = []
    after = []
    for batch in tqdm(batches, desc="Computing paired latent metrics"):
        target_batch = make_shifted_batch(batch, normalizer, appearance_shift)
        source_states, _actions, _locations = batch_to_device_time_major(batch, device)
        target_states, _target_actions, _target_locations = batch_to_device_time_major(
            target_batch,
            device,
        )
        source_z = encode_states(model, source_states)
        target_z = encode_states(model, target_states)
        adapted_z = adapter(target_z)
        t = min(source_z.shape[0], target_z.shape[0])
        before.append(F.mse_loss(target_z[:t], source_z[:t]).item())
        after.append(F.mse_loss(adapted_z[:t], source_z[:t]).item())

    return {
        "paired_latent_mse_before_adapter": sum(before) / len(before),
        "paired_latent_mse_after_adapter": sum(after) / len(after),
    }


@torch.no_grad()
def rollout_metrics(
    model: torch.nn.Module,
    adapter: AppearanceAdapter,
    batches,
    normalizer,
    appearance_shift: str,
    device: torch.device,
):
    losses = []
    source_losses = []
    persistence_losses = []

    for batch in tqdm(batches, desc="Evaluating adapted rollouts"):
        target_batch = make_shifted_batch(batch, normalizer, appearance_shift)
        source_states, actions, _locations = batch_to_device_time_major(batch, device)
        target_states, _target_actions, _target_locations = batch_to_device_time_major(
            target_batch,
            device,
        )
        source_z = encode_states(model, source_states)
        target_z = encode_states(model, target_states)
        adapted_z = adapter(target_z)

        steps = min(actions.shape[0], adapted_z.shape[0] - 1)
        predictions = model.level1.forward_prior(
            adapted_z[0],
            repr_input=True,
            actions=actions[:steps],
            T=steps,
        ).pred_output.predictions

        t = min(predictions.shape[0], adapted_z.shape[0], source_z.shape[0])
        losses.append(
            F.mse_loss(
                predictions[1:t],
                adapted_z[1:t],
                reduction="none",
            )
            .flatten(1)
            .mean(dim=1)
        )
        source_losses.append(
            F.mse_loss(
                predictions[1:t],
                source_z[1:t],
                reduction="none",
            )
            .flatten(1)
            .mean(dim=1)
        )
        persistence = adapted_z[0].unsqueeze(0).expand_as(adapted_z[1:t])
        persistence_losses.append(
            F.mse_loss(
                persistence,
                adapted_z[1:t],
                reduction="none",
            )
            .flatten(1)
            .mean(dim=1)
        )

    pred = torch.stack(losses).mean(dim=0)
    source_pred = torch.stack(source_losses).mean(dim=0)
    persistence = torch.stack(persistence_losses).mean(dim=0)

    return {
        "adapted_rollout_mse_by_horizon": [x.item() for x in pred],
        "adapted_rollout_to_source_mse_by_horizon": [x.item() for x in source_pred],
        "adapted_persistence_mse_by_horizon": [x.item() for x in persistence],
        "adapted_rollout_vs_persistence_mse_ratio": (
            pred.mean() / persistence.mean()
        ).item(),
    }


def load_adapter_for_eval(
    config: AdapterEvalConfig,
    latent_dim: int,
    device: torch.device,
) -> AppearanceAdapter:
    config.adapter.latent_dim = latent_dim
    adapter = AppearanceAdapter(config.adapter).to(device)
    checkpoint = torch.load(config.adapter_checkpoint_path, map_location=device)
    adapter.load_state_dict(checkpoint["adapter_state_dict"], strict=True)
    adapter.eval()
    return adapter


def evaluate_adapter(config: AdapterEvalConfig):
    seed_everything(config.seed)
    device = resolve_device(config.data.device)

    baseline_config, train_loader, val_loader = build_dataloaders(
        source_config_path=config.source_config_path,
        data_config=config.data,
        val_batches=max(config.probe_val_batches, config.rollout_batches),
    )
    sample = next(iter(train_loader))
    model = build_source_model(
        baseline_config=baseline_config,
        sample=sample,
        checkpoint_path=config.source_checkpoint_path,
        device=device,
    )
    freeze_source_model(model, freeze_backbone=True, freeze_predictor=True)
    adapter = load_adapter_for_eval(config, model.level1.repr_dim, device)

    train_batches = take_batches(
        train_loader,
        config.probe_train_batches,
        "Collecting source probe train batches",
    )
    val_batches = take_batches(
        val_loader,
        max(config.probe_val_batches, config.rollout_batches),
        "Collecting eval batches",
    )
    shifted_val_batches = [
        make_shifted_batch(batch, val_loader.normalizer, config.data.appearance_shift)
        for batch in val_batches
    ]

    source_train_z, source_train_loc = encode_locations(
        model,
        train_batches,
        device=device,
        adapter=None,
    )
    target_z, target_loc = encode_locations(
        model,
        shifted_val_batches[: config.probe_val_batches],
        device=device,
        adapter=None,
    )
    adapted_target_z, adapted_target_loc = encode_locations(
        model,
        shifted_val_batches[: config.probe_val_batches],
        device=device,
        adapter=adapter,
    )

    report = {
        "source_config_path": config.source_config_path,
        "source_checkpoint_path": config.source_checkpoint_path,
        "adapter_checkpoint_path": config.adapter_checkpoint_path,
        "appearance_shift": config.data.appearance_shift,
        "probe_train_samples": int(source_train_z.shape[0]),
        "probe_val_samples": int(target_z.shape[0]),
        **adapter.delta_stats(),
    }
    report.update(
        {
            f"unadapted_{key}": value
            for key, value in train_linear_probe(
                source_train_z,
                source_train_loc,
                target_z,
                target_loc,
                train_loader.normalizer,
                config.probe_steps,
            ).items()
        }
    )
    report.update(
        {
            f"adapted_{key}": value
            for key, value in train_linear_probe(
                source_train_z,
                source_train_loc,
                adapted_target_z,
                adapted_target_loc,
                train_loader.normalizer,
                config.probe_steps,
            ).items()
        }
    )
    report.update(
        paired_latent_metrics(
            model=model,
            adapter=adapter,
            batches=val_batches[: config.probe_val_batches],
            normalizer=val_loader.normalizer,
            appearance_shift=config.data.appearance_shift,
            device=device,
        )
    )
    report.update(
        rollout_metrics(
            model=model,
            adapter=adapter,
            batches=val_batches[: config.rollout_batches],
            normalizer=val_loader.normalizer,
            appearance_shift=config.data.appearance_shift,
            device=device,
        )
    )

    output_path = Path(config.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(jsonable(report), indent=2) + "\n")
    print(json.dumps(jsonable(report), indent=2))
    return report


class WhiteHole(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        adapter: Optional[AppearanceAdapter] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.appearance_adapt = adapter or AppearanceAdapter(AppearanceAdapterConfig())

    def forward(self, x, *args, **kwargs):
        z = self.encoder(x)
        z = self.appearance_adapt(z)
        return self.predictor(z, *args, **kwargs)
