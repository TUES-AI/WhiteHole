"""MoVie-style shallow spatial adaptation for the LeWM ViT encoder."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPooling


class AffineSpatialTransformer(nn.Module):
    """Predict and apply one identity-initialized affine warp per image."""

    def __init__(self, in_channels: int, hidden_channels: int = 32):
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 6),
        )
        final = self.regressor[-1]
        nn.init.zeros_(final.weight)
        with torch.no_grad():
            final.bias.copy_(
                torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            )

    def affine(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.localization(x)).reshape(-1, 2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        theta = self.affine(x)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )


class MoVieViTEncoder(nn.Module):
    """Apply STNs at RGB input and on the ViT patch-projection grid."""

    def __init__(self, base_encoder: nn.Module, hidden_channels: int = 32):
        super().__init__()
        self.base_encoder = base_encoder
        feature_channels = int(base_encoder.config.hidden_size)
        self.input_stn = AffineSpatialTransformer(3, hidden_channels)
        self.feature_stn = AffineSpatialTransformer(
            feature_channels, hidden_channels
        )
        self.config = base_encoder.config

    def stn_parameters(self):
        yield from self.input_stn.parameters()
        yield from self.feature_stn.parameters()

    def encoder_parameters(self):
        yield from self.base_encoder.parameters()

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        bool_masked_pos: torch.BoolTensor | None = None,
        head_mask: torch.Tensor | None = None,
        interpolate_pos_encoding: bool | None = None,
        **_kwargs: Any,
    ) -> BaseModelOutputWithPooling:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        base = self.base_encoder
        expected_dtype = base.embeddings.patch_embeddings.projection.weight.dtype
        pixel_values = pixel_values.to(expected_dtype)
        pixel_values = self.input_stn(pixel_values)

        batch_size, _channels, height, width = pixel_values.shape
        patch_grid = base.embeddings.patch_embeddings.projection(pixel_values)
        patch_grid = self.feature_stn(patch_grid)
        embeddings = patch_grid.flatten(2).transpose(1, 2)

        if bool_masked_pos is not None:
            mask_tokens = base.embeddings.mask_token.expand(
                batch_size, embeddings.shape[1], -1
            )
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        cls_tokens = base.embeddings.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        if interpolate_pos_encoding:
            position = base.embeddings.interpolate_pos_encoding(
                embeddings, height, width
            )
        else:
            position = base.embeddings.position_embeddings
        embeddings = base.embeddings.dropout(embeddings + position)

        prepared_head_mask = base.get_head_mask(
            head_mask, base.config.num_hidden_layers
        )
        encoder_outputs = base.encoder(
            embeddings, head_mask=prepared_head_mask
        )
        sequence_output = base.layernorm(encoder_outputs.last_hidden_state)
        pooled_output = (
            base.pooler(sequence_output) if base.pooler is not None else None
        )
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


def build_movie_encoder(
    base_encoder: nn.Module, config: dict | None = None
) -> MoVieViTEncoder:
    config = config or {}
    return MoVieViTEncoder(
        base_encoder,
        hidden_channels=int(config.get("stn_hidden_channels", 32)),
    )
