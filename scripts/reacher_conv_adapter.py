"""Small visual adapters for LeWM Reacher experiments."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn


class SmallConvAdapter(nn.Module):
    """Tiny identity-initialized residual CNN for normalized RGB observations."""

    def __init__(
        self,
        channels: int = 16,
        depth: int = 2,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        layers: list[nn.Module] = [
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.GELU(),
        ]
        for _ in range(depth - 1):
            layers.extend(
                [
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                    nn.GELU(),
                ]
            )
        final = nn.Conv2d(channels, 3, kernel_size=3, padding=1)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)

        self.net = nn.Sequential(*layers)
        self.residual_scale = float(residual_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual_scale * self.net(x)


class CoordUNetAdapter(nn.Module):
    """Small coordinate-aware encoder-decoder with identity residual output."""

    def __init__(self, base_channels: int = 16, residual_scale: float = 1.0):
        super().__init__()
        c = base_channels
        self.enc0 = nn.Conv2d(5, c, kernel_size=3, padding=1)
        self.enc1 = nn.Conv2d(c, 2 * c, kernel_size=3, stride=2, padding=1)
        self.enc2 = nn.Conv2d(2 * c, 3 * c, kernel_size=3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(3 * c, 4 * c, kernel_size=3, stride=2, padding=1)
        self.global_context = nn.Linear(4 * c, 4 * c)
        self.dec2 = nn.Conv2d(4 * c, 3 * c, kernel_size=3, padding=1)
        self.dec1 = nn.Conv2d(3 * c, 2 * c, kernel_size=3, padding=1)
        self.dec0 = nn.Conv2d(2 * c, c, kernel_size=3, padding=1)
        self.final = nn.Conv2d(c, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)
        self.residual_scale = float(residual_scale)

    @staticmethod
    def _coordinates(x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype),
            torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        return torch.stack((xx, yy)).unsqueeze(0).expand(len(x), -1, -1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = F.gelu(self.enc0(torch.cat((x, self._coordinates(x)), dim=1)))
        e1 = F.gelu(self.enc1(e0))
        e2 = F.gelu(self.enc2(e1))
        z = F.gelu(self.enc3(e2))
        context = self.global_context(z.mean(dim=(-2, -1))).unsqueeze(-1).unsqueeze(-1)
        z = z + context
        z = F.interpolate(z, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        z = F.gelu(self.dec2(z) + e2)
        z = F.interpolate(z, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        z = F.gelu(self.dec1(z) + e1)
        z = F.interpolate(z, size=e0.shape[-2:], mode="bilinear", align_corners=False)
        z = F.gelu(self.dec0(z) + e0)
        return x + self.residual_scale * self.final(z)


def build_input_adapter(config: dict) -> nn.Module:
    architecture = config.get("architecture", "conv")
    if architecture == "conv":
        return SmallConvAdapter(
            channels=int(config.get("channels", 16)),
            depth=int(config.get("depth", 2)),
            residual_scale=float(config.get("residual_scale", 1.0)),
        )
    if architecture == "coord_unet":
        return CoordUNetAdapter(
            base_channels=int(config.get("base_channels", 16)),
            residual_scale=float(config.get("residual_scale", 1.0)),
        )
    raise ValueError(f"Unknown input adapter architecture: {architecture!r}")


class AdaptedLeWM(nn.Module):
    """Apply an observation adapter before a frozen LeWM model."""

    def __init__(
        self,
        base_model: nn.Module,
        adapter: nn.Module,
        keys: Iterable[str] = ("pixels", "goal"),
    ):
        super().__init__()
        self.base_model = base_model
        self.adapter = adapter
        self.keys = tuple(keys)

    def _adapt_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim < 4:
            return tensor
        leading = tensor.shape[:-3]
        flat = tensor.reshape(-1, *tensor.shape[-3:])
        adapted = self.adapter(flat)
        return adapted.reshape(*leading, *adapted.shape[-3:])

    def _adapt_info(self, info: dict) -> dict:
        out = dict(info)
        for key in self.keys:
            value = out.get(key)
            if torch.is_tensor(value):
                out[key] = self._adapt_tensor(value)
        return out

    def encode(self, info: dict) -> dict:
        return self.base_model.encode(self._adapt_info(info))

    def get_cost(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        # CEM expands identical observations across its candidate dimension.
        # Adapt one copy, then broadcast it instead of running the CNN hundreds
        # of times. LeWM caches these encoded initial/goal embeddings for later
        # CEM iterations in the same solve.
        if "emb" in info_dict and "goal_emb" in info_dict:
            return self.base_model.get_cost(info_dict, action_candidates)

        adapted = dict(info_dict)
        for key in self.keys:
            value = adapted.get(key)
            if not torch.is_tensor(value) or value.ndim < 5:
                continue
            unique = value[:, 0]
            adapted_unique = self._adapt_tensor(unique)
            adapted[key] = adapted_unique.unsqueeze(1).expand_as(value)

        costs = self.base_model.get_cost(adapted, action_candidates)
        for key in ("emb", "goal_emb"):
            if key in adapted:
                info_dict[key] = adapted[key]
        return costs
