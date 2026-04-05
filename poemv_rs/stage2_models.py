from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Stage2DNNConfig:
    obs_dim: int = 9
    hidden: int = 64
    correction_scale: float = 0.25
    use_residual_boundary: bool = True


class BoundaryCorrectionNet(nn.Module):
    def __init__(self, cfg: Stage2DNNConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = nn.Sequential(
            nn.Linear(cfg.obs_dim, cfg.hidden),
            nn.Tanh(),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.Tanh(),
        )
        self.head_lower = nn.Linear(cfg.hidden, 2)
        self.head_upper = nn.Linear(cfg.hidden, 2)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor):
        h = self.backbone(obs)
        raw_lower = self.head_lower(h)
        raw_upper = self.head_upper(h)
        lower_scale = torch.exp(self.cfg.correction_scale * raw_lower)
        upper_scale = torch.exp(self.cfg.correction_scale * raw_upper)
        return lower_scale, upper_scale


class DirectBoundaryNet(nn.Module):
    """
    Input:
        obs = [tau, x, p, w_cur0, w_cur1, center0, center1, diff0, diff1]

    Output:
        positive lower_gap, upper_gap in weight space

    Boundary:
        lower = center - lower_gap
        upper = center + upper_gap
    """
    def __init__(self, cfg: Stage2DNNConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = nn.Sequential(
            nn.Linear(cfg.obs_dim, 20),
            nn.LeakyReLU(0.2),
            nn.Linear(20, 40),
            nn.LeakyReLU(0.2),
            nn.Linear(40, 80),
            nn.LeakyReLU(0.2),
        )
        self.head_lower_gap = nn.Linear(80, 2)
        self.head_upper_gap = nn.Linear(80, 2)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor):
        h = self.backbone(obs)
        raw_l = self.head_lower_gap(h)
        raw_u = self.head_upper_gap(h)
        # keep positivity for lower/upper distance from center
        lower_gap = torch.nn.functional.softplus(raw_l) + 1e-6
        upper_gap = torch.nn.functional.softplus(raw_u) + 1e-6
        return lower_gap, upper_gap