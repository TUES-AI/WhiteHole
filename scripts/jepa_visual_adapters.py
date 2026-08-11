"""Small identity-initialized image adapters and structured visual shifts."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

SHIFT_NAMES = ("source", "rbg", "affine", "composed")


def _coordinates(x: torch.Tensor) -> torch.Tensor:
    height, width = x.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype),
        torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    return torch.stack((xx, yy)).unsqueeze(0).expand(len(x), -1, -1, -1)


class CoordUNetImageAdapter(nn.Module):
    """Coordinate-aware residual U-Net with exact identity initialization."""

    def __init__(self, base_channels: int = 16):
        super().__init__()
        c = base_channels
        self.enc0 = nn.Conv2d(5, c, 3, padding=1)
        self.enc1 = nn.Conv2d(c, 2 * c, 3, stride=2, padding=1)
        self.enc2 = nn.Conv2d(2 * c, 3 * c, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(3 * c, 4 * c, 3, stride=2, padding=1)
        self.global_context = nn.Linear(4 * c, 4 * c)
        self.dec2 = nn.Conv2d(4 * c, 3 * c, 3, padding=1)
        self.dec1 = nn.Conv2d(3 * c, 2 * c, 3, padding=1)
        self.dec0 = nn.Conv2d(2 * c, c, 3, padding=1)
        self.final = nn.Conv2d(c, 3, 3, padding=1)
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = F.gelu(self.enc0(torch.cat((x, _coordinates(x)), dim=1)))
        e1 = F.gelu(self.enc1(e0))
        e2 = F.gelu(self.enc2(e1))
        z = F.gelu(self.enc3(e2))
        context = self.global_context(z.mean((-2, -1)))[:, :, None, None]
        z = z + context
        z = F.interpolate(z, e2.shape[-2:], mode="bilinear", align_corners=False)
        z = F.gelu(self.dec2(z) + e2)
        z = F.interpolate(z, e1.shape[-2:], mode="bilinear", align_corners=False)
        z = F.gelu(self.dec1(z) + e1)
        z = F.interpolate(z, e0.shape[-2:], mode="bilinear", align_corners=False)
        z = F.gelu(self.dec0(z) + e0)
        return x + self.final(z)


class GridColorImageAdapter(nn.Module):
    """Predict a smooth dense warp, global color map, and small RGB residual."""

    def __init__(self, max_displacement: float = 0.75):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(5, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.context = nn.Linear(64, 64)
        self.flow_head = nn.Conv2d(64, 2, 3, padding=1)
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)
        self.color = nn.Conv2d(3, 3, 1)
        with torch.no_grad():
            self.color.weight.zero_()
            self.color.weight[:, :, 0, 0].copy_(torch.eye(3))
            self.color.bias.zero_()
        self.residual = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 3, 3, padding=1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.max_displacement = float(max_displacement)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.trunk(torch.cat((x, _coordinates(x)), dim=1))
        context = self.context(features.mean((-2, -1)))[:, :, None, None]
        flow = self.flow_head(features + context)
        flow = F.interpolate(flow, x.shape[-2:], mode="bilinear", align_corners=False)
        flow = self.max_displacement * torch.tanh(flow)
        grid = _coordinates(x).permute(0, 2, 3, 1) + flow.permute(0, 2, 3, 1)
        warped = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=True,
        )
        corrected = self.color(warped)
        return corrected + self.residual(corrected)


def build_image_adapter(name: str) -> nn.Module:
    if name == "unet":
        return CoordUNetImageAdapter()
    if name == "grid_color":
        return GridColorImageAdapter()
    raise ValueError(f"Unknown image adapter: {name!r}")


def _affine_parameters(indices: torch.Tensor, seed: int) -> tuple[torch.Tensor, ...]:
    """Return one fixed target-domain rotation and shear for every sample."""
    del seed
    count = len(indices)
    return (
        torch.full((count,), 30.0),
        torch.full((count,), 18.0),
        torch.full((count,), 8.0),
    )


def apply_visual_shift(
    images: torch.Tensor,
    shift: str,
    indices: torch.Tensor,
    seed: int = 20260811,
) -> torch.Tensor:
    """Apply a deterministic structured shift to raw [0, 1] RGB tensors."""
    if shift not in SHIFT_NAMES:
        raise ValueError(f"Unknown shift {shift!r}; expected one of {SHIFT_NAMES}")
    output = images
    if shift in ("rbg", "composed"):
        output = output[:, (0, 2, 1)]
    if shift not in ("affine", "composed"):
        return output

    angle, shear_x, shear_y = _affine_parameters(indices.cpu(), seed)
    angle = torch.deg2rad(angle).to(images.device, images.dtype)
    shear_x = torch.tan(torch.deg2rad(shear_x)).to(images.device, images.dtype)
    shear_y = torch.tan(torch.deg2rad(shear_y)).to(images.device, images.dtype)
    cos, sin = torch.cos(angle), torch.sin(angle)
    shear = torch.zeros(len(images), 2, 2, device=images.device, dtype=images.dtype)
    shear[:, 0, 0] = 1.0
    shear[:, 0, 1] = shear_x
    shear[:, 1, 0] = shear_y
    shear[:, 1, 1] = 1.0
    rotation = torch.zeros_like(shear)
    rotation[:, 0, 0] = cos
    rotation[:, 0, 1] = -sin
    rotation[:, 1, 0] = sin
    rotation[:, 1, 1] = cos
    matrix = rotation @ shear
    theta = torch.zeros(len(images), 2, 3, device=images.device, dtype=images.dtype)
    theta[:, :, :2] = matrix
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    return F.grid_sample(
        output,
        grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=False,
    )
