"""Neural C-SDF and PPO actor-critic models."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Normal


def periodic_features(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((torch.sin(q), torch.cos(q)), dim=-1)


class NeuralCSDF(nn.Module):
    """Predict signed capsule clearance and collision probability from q."""

    def __init__(self, hidden_dims: tuple[int, ...] = (256, 256, 128)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = 12
        for hidden in hidden_dims:
            layers.extend((nn.Linear(input_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden)))
            input_dim = hidden
        self.backbone = nn.Sequential(*layers)
        self.clearance_head = nn.Linear(input_dim, 1)
        self.collision_head = nn.Linear(input_dim, 1)

    def forward(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if q.shape[-1] != 6:
            raise ValueError("q must end in six joint dimensions")
        hidden = self.backbone(periodic_features(q))
        return self.clearance_head(hidden).squeeze(-1), self.collision_head(hidden).squeeze(-1)


class TrajectoryActorCritic(nn.Module):
    """Gaussian residual-control policy with a scalar critic."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 384,
        initial_std: float = 0.35,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), math.log(initial_std)))
        nn.init.zeros_(self.actor_mean.weight)
        nn.init.constant_(self.actor_mean.bias, 0.5)

    def distribution_value(self, state: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        hidden = self.encoder(state)
        mean = torch.tanh(self.actor_mean(hidden))
        std = torch.exp(self.log_std).clamp(0.03, 1.0).expand_as(mean)
        return Normal(mean, std), self.critic(hidden).squeeze(-1)

    def act(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, value = self.distribution_value(state)
        action = distribution.mean if deterministic else distribution.rsample()
        action = action.clamp(-1.0, 1.0)
        log_prob = distribution.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def evaluate_actions(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, value = self.distribution_value(state)
        return (
            distribution.log_prob(action).sum(dim=-1),
            distribution.entropy().sum(dim=-1),
            value,
        )
