from typing import Literal

import torch
import torch.nn.functional as F


AppearanceShift = Literal["source", "mild", "medium", "hard"]
APPEARANCE_SHIFTS = ("source", "mild", "medium", "hard")


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


def _filter_channel(channel: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    leading_shape = channel.shape[:-2]
    height, width = channel.shape[-2:]
    images = channel.reshape(-1, 1, height, width)
    kernel = kernel.to(device=channel.device, dtype=channel.dtype).view(
        1,
        1,
        *kernel.shape,
    )
    filtered = F.conv2d(images, kernel, padding=kernel.shape[-1] // 2)
    return filtered.reshape(*leading_shape, height, width)


def _blur_channel(channel: torch.Tensor) -> torch.Tensor:
    kernel = channel.new_tensor(
        [
            [1.0, 2.0, 1.0],
            [2.0, 4.0, 2.0],
            [1.0, 2.0, 1.0],
        ]
    ) / 16.0
    return _filter_channel(channel, kernel)


def _max_pool_channel(channel: torch.Tensor, kernel_size: int) -> torch.Tensor:
    leading_shape = channel.shape[:-2]
    height, width = channel.shape[-2:]
    images = channel.reshape(-1, 1, height, width)
    pooled = F.max_pool2d(images, kernel_size, stride=1, padding=kernel_size // 2)
    return pooled.reshape(*leading_shape, height, width)


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
    elif shift == "medium":
        shifted_agent = 0.35 * walls
        shifted_walls = agent
    else:
        texture = _static_texture_like(obs).view(
            *([1] * (obs.ndim - 3)), *obs.shape[-2:]
        )
        stripe = texture + 0.5
        agent_blur = _blur_channel(agent)
        agent_halo = (_max_pool_channel(agent, 5) - agent).clamp_min(0.0)
        walls_blur = _blur_channel(walls)
        walls_thick = _max_pool_channel(walls, 3)
        wall_edges = (walls_thick - walls_blur).abs()

        shifted_agent = (
            0.35 * walls_thick * (0.35 + 0.65 * stripe)
            + 0.25 * wall_edges
            + 0.10 * agent_halo
            + (18.0 / 255.0 * scale) * texture
        )
        shifted_walls = (
            0.65 * agent_blur
            + 0.20 * agent
            + 0.10 * walls_blur
            + (14.0 / 255.0 * scale) * texture
        )

    shifted = torch.stack([shifted_agent, shifted_walls], dim=-3).clamp(0, scale)
    if original_dtype.is_floating_point:
        return shifted.to(original_dtype)
    return shifted.round().to(original_dtype)
