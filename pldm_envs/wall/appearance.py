from typing import Literal

import torch


AppearanceShift = Literal["source", "mild", "medium"]
APPEARANCE_SHIFTS = ("source", "mild", "medium")


def _static_texture_like(obs: torch.Tensor) -> torch.Tensor:
    height, width = obs.shape[-2:]
    y = torch.arange(height, device=obs.device).view(height, 1)
    x = torch.arange(width, device=obs.device).view(1, width)
    texture = ((x * 13 + y * 17) % 23).to(torch.float32) / 22.0
    return texture - 0.5


def _intensity_scale(obs: torch.Tensor) -> float:
    if obs.dtype.is_floating_point and obs.detach().amax().item() <= 2.0:
        return 1.0
    return 255.0


def apply_appearance_shift(
    obs: torch.Tensor,
    shift: AppearanceShift = "source",
) -> torch.Tensor:
    """Apply appearance-only changes to a two-channel wall observation.

    The transform preserves the underlying state, transition dynamics, and action
    semantics. It only changes how the two rendered channels look to the encoder.
    """

    if shift == "source":
        return obs.clone()
    if shift not in APPEARANCE_SHIFTS:
        raise ValueError(f"Unknown appearance shift: {shift}")
    if obs.shape[-3] != 2:
        raise ValueError(f"Expected a 2-channel observation, got shape {obs.shape}")

    original_dtype = obs.dtype
    obs_f = obs.to(torch.float32)
    scale = _intensity_scale(obs_f)
    agent = obs_f[..., 0, :, :]
    walls = obs_f[..., 1, :, :]

    if shift == "mild":
        texture = _static_texture_like(obs).view(
            *([1] * (obs.ndim - 3)), *obs.shape[-2:]
        )
        shifted_agent = 0.60 * agent + (8.0 / 255.0 * scale) * texture
        shifted_walls = 1.30 * walls + (8.0 / 255.0 * scale) * texture
    else:
        shifted_agent = 0.35 * walls
        shifted_walls = agent

    shifted = torch.stack([shifted_agent, shifted_walls], dim=-3).clamp(0, scale)
    if original_dtype.is_floating_point:
        return shifted.to(original_dtype)
    return shifted.round().to(original_dtype)
