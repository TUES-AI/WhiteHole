"""Small visual adapters for LeWM Reacher experiments."""

from __future__ import annotations

from collections.abc import Iterable

import torch
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
        return self.base_model.get_cost(
            self._adapt_info(info_dict), action_candidates
        )
