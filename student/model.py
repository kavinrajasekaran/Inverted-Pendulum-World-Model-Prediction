"""Student world model.

The official evaluator only calls ``initial_hidden`` and ``forward`` through the
locked prediction helper.  This implementation keeps that interface, but uses a
small physics prior for the MuJoCo state layout:

    observation = [cart_position, pole_angle, cart_velocity, pole_angular_velocity]

Training starts by fitting a linear residual-dynamics model from the first batch.
That closed-form base is very sharp for the small-angle pendulum regime.  The
neural network then learns a bounded correction on top, using rollout loss to
improve open-loop behavior without destroying the system-identification base.
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.net(x)


class StudentWorldModel(nn.Module):
    def __init__(
        self,
        obs_dim: int = 4,
        act_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 2,
        use_gru: bool = False,
        delta_limit: float = 1.0,
    ):
        super().__init__()
        if obs_dim != 4:
            raise ValueError("This homework model expects the 4D InvertedPendulum observation.")
        self.use_gru = bool(use_gru)
        self.delta_limit = float(delta_limit)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.dt = 0.04

        # These buffers are filled from the training normalizer by
        # student.losses.compute_loss and are saved in the checkpoint state_dict.
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))
        self.register_buffer("act_mean", torch.zeros(act_dim))
        self.register_buffer("act_std", torch.ones(act_dim))
        self.register_buffer("delta_mean", torch.zeros(obs_dim))
        self.register_buffer("delta_std", torch.ones(obs_dim))
        self.register_buffer("linear_initialized", torch.zeros((), dtype=torch.bool))
        self.register_buffer("state_bounds", torch.tensor([1.05, 0.18, 1.0, 1.0], dtype=torch.float32))

        feature_dim = obs_dim + act_dim + 5
        self.input = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(int(num_layers))])
        self.gru = nn.GRUCell(hidden_dim, hidden_dim) if self.use_gru else None
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, obs_dim),
        )
        self.linear_delta = nn.Linear(obs_dim + act_dim, obs_dim)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        nn.init.zeros_(self.linear_delta.weight)
        nn.init.zeros_(self.linear_delta.bias)

    def set_normalizer(self, normalizer) -> None:
        """Copy train-set normalization statistics into checkpointed buffers."""
        device = self.obs_mean.device
        dtype = self.obs_mean.dtype
        self.obs_mean.copy_(torch.as_tensor(normalizer.obs_mean, device=device, dtype=dtype))
        self.obs_std.copy_(torch.as_tensor(normalizer.obs_std, device=device, dtype=dtype))
        self.act_mean.copy_(torch.as_tensor(normalizer.act_mean, device=device, dtype=dtype))
        self.act_std.copy_(torch.as_tensor(normalizer.act_std, device=device, dtype=dtype))
        self.delta_mean.copy_(torch.as_tensor(normalizer.delta_mean, device=device, dtype=dtype))
        self.delta_std.copy_(torch.as_tensor(normalizer.delta_std, device=device, dtype=dtype))

    @torch.no_grad()
    def initialize_linear_dynamics(self, states: torch.Tensor, actions: torch.Tensor, ridge: float = 1e-5) -> None:
        """Fit and freeze the linear normalized-delta path from a training batch."""
        if bool(self.linear_initialized.item()):
            return
        obs = states[:, :-1].reshape(-1, self.obs_dim)
        act = actions.reshape(-1, self.act_dim)
        target_delta = (states[:, 1:] - states[:, :-1]).reshape(-1, self.obs_dim)
        obs_norm = (obs - self.obs_mean) / self.obs_std.clamp_min(1e-6)
        act_norm = (act - self.act_mean) / self.act_std.clamp_min(1e-6)
        target_norm = (target_delta - self.delta_mean) / self.delta_std.clamp_min(1e-6)
        features = torch.cat([obs_norm, act_norm, torch.ones(obs_norm.shape[0], 1, device=obs_norm.device)], dim=-1)
        eye = torch.eye(features.shape[-1], dtype=features.dtype, device=features.device)
        coeff = torch.linalg.solve(features.T @ features + float(ridge) * eye, features.T @ target_norm)
        self.linear_delta.weight.copy_(coeff[:-1].T)
        self.linear_delta.bias.copy_(coeff[-1])
        self.linear_delta.weight.requires_grad_(False)
        self.linear_delta.bias.requires_grad_(False)
        self.linear_initialized.fill_(True)

    def initial_hidden(self, batch_size: int, device: torch.device):
        if not self.use_gru:
            return None
        return torch.zeros(batch_size, self.gru.hidden_size, device=device)

    def forward(self, obs_norm: torch.Tensor, act_norm: torch.Tensor, hidden=None):
        obs_raw = obs_norm * self.obs_std + self.obs_mean
        act_raw = act_norm * self.act_std + self.act_mean
        theta = obs_raw[:, 1:2]
        vel = obs_raw[:, 2:4]
        raw_features = torch.cat(
            [
                torch.sin(theta),
                torch.cos(theta) - 1.0,
                vel,
                act_raw,
            ],
            dim=-1,
        )
        base = torch.cat([obs_norm, act_norm], dim=-1)
        feat = self.input(torch.cat([base, raw_features], dim=-1))
        for block in self.blocks:
            feat = block(feat)
        if self.gru is not None:
            if hidden is None:
                hidden = self.initial_hidden(obs_norm.shape[0], obs_norm.device)
            hidden = self.gru(feat, hidden)
            feat = hidden
        base_delta = self.linear_delta(base)
        residual_delta = self.delta_limit * torch.tanh(self.head(feat) / self.delta_limit)
        delta_norm = base_delta + residual_delta
        delta_raw = delta_norm * self.delta_std + self.delta_mean
        next_raw = torch.clamp(obs_raw + delta_raw, min=-self.state_bounds, max=self.state_bounds)
        guarded_delta = next_raw - obs_raw
        guarded_delta_norm = (guarded_delta - self.delta_mean) / self.delta_std.clamp_min(1e-6)
        return guarded_delta_norm, hidden
