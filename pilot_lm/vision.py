from __future__ import annotations

import torch
import torch.nn as nn


class VisionToPilotAdapter(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        pilot_dim: int,
        n_visual_tokens: int = 8,
        hidden_mult: int = 2,
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.pilot_dim = pilot_dim
        self.n_visual_tokens = n_visual_tokens
        hidden_dim = max(pilot_dim, vision_dim) * hidden_mult
        self.net = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_visual_tokens * pilot_dim),
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        visual_tokens = self.net(image_features)
        return visual_tokens.view(image_features.size(0), self.n_visual_tokens, self.pilot_dim)
