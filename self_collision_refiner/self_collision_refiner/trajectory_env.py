"""Vectorized residual trajectory-refinement environment for PPO."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from .geometry import cubic_interpolation_matrix, unwrap_trajectory
from .models import NeuralCSDF, periodic_features


class TrajectoryRefinementEnv:
    """A short-horizon MDP that deforms all interior trajectory control points."""

    def __init__(
        self,
        source_paths: np.ndarray,
        csdf_model: NeuralCSDF,
        config: dict,
        device: str,
        seed: int,
    ) -> None:
        source = np.asarray(source_paths, dtype=np.float32)
        if source.ndim != 3 or source.shape[1:] != (64, 6):
            raise ValueError("source_paths must have shape [N, 64, 6]")
        self.source_paths = source
        self.csdf_model = csdf_model.eval()
        for parameter in self.csdf_model.parameters():
            parameter.requires_grad_(False)
        self.device = device
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.control_count = int(config["control_points"])
        self.output_points = int(config["output_points"])
        self.episode_steps = int(config["episode_steps"])
        self.action_step_rad = float(config["action_step_rad"])
        self.max_residual_rad = float(config["max_residual_rad"])
        self.initial_corruption_rad = float(config["initial_corruption_rad"])
        self.margin_m = float(config["required_clearance_m"])
        self.gamma = float(config["gamma"])
        self.weights = dict(config["reward_weights"])
        self.proxy_scales = dict(config.get("proxy_scales", {}))
        self.derivative_reference_points = int(
            config.get("derivative_reference_points", self.output_points)
        )
        if self.output_points < 4 or self.derivative_reference_points < 4:
            raise ValueError("output_points and derivative_reference_points must be at least four")
        self.control_indices = np.linspace(0, 63, self.control_count).round().astype(int)
        output_control_indices = np.linspace(0, self.output_points - 1, self.control_count)
        self.output_control_indices = np.round(output_control_indices).astype(int)
        self.basis = torch.as_tensor(
            cubic_interpolation_matrix(self.control_count, self.output_points),
            dtype=torch.float32,
            device=device,
        )
        self.derivative_basis = (
            self.basis
            if self.derivative_reference_points == self.output_points
            else torch.as_tensor(
                cubic_interpolation_matrix(
                    self.control_count, self.derivative_reference_points
                ),
                dtype=torch.float32,
                device=device,
            )
        )
        self.residual_dim = (self.control_count - 2) * 6
        self.action_dim = self.control_count - 2
        self.state_dim = self.control_count * 12 * 2 + self.control_count + self.residual_dim + 4
        self.original_controls: torch.Tensor | None = None
        self.residual: torch.Tensor | None = None
        self.current_step = 0
        self.previous_potential: torch.Tensor | None = None
        self.last_metrics: dict[str, torch.Tensor] = {}

    def _controls_from_paths(self, paths: np.ndarray) -> torch.Tensor:
        controls = np.stack(
            [unwrap_trajectory(path)[self.control_indices] for path in paths], axis=0
        )
        return torch.as_tensor(controls, dtype=torch.float32, device=self.device)

    def _paths(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.original_controls is None or self.residual is None:
            raise RuntimeError("reset must be called first")
        controls = self.original_controls.clone()
        controls[:, 1:-1] += self.residual
        continuous = torch.einsum("tc,bcj->btj", self.basis, controls)
        continuous[:, 0] = self.original_controls[:, 0]
        continuous[:, -1] = self.original_controls[:, -1]
        wrapped = torch.remainder(continuous + torch.pi, 2.0 * torch.pi) - torch.pi
        return continuous, wrapped

    def _safety_direction(self) -> torch.Tensor:
        """Differentiate a soft minimum C-SDF clearance with respect to controls."""
        with torch.enable_grad():
            residual = self.residual.detach().clone().requires_grad_(True)
            controls = self.original_controls.detach().clone()
            controls[:, 1:-1] += residual
            continuous = torch.einsum("tc,bcj->btj", self.basis, controls)
            continuous[:, 0] = self.original_controls[:, 0]
            continuous[:, -1] = self.original_controls[:, -1]
            wrapped = torch.remainder(continuous + torch.pi, 2.0 * torch.pi) - torch.pi
            clearance, _ = self.csdf_model(wrapped.reshape(-1, 6))
            clearance = clearance.reshape(len(wrapped), self.output_points)
            temperature = 0.004
            soft_minimum = -temperature * torch.logsumexp(
                -clearance / temperature, dim=1
            )
            gradient = torch.autograd.grad(soft_minimum.sum(), residual)[0]
        norm = torch.linalg.norm(gradient, dim=-1, keepdim=True).clamp_min(1.0e-8)
        return (gradient / norm).detach()

    @torch.no_grad()
    def _measure(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        continuous, wrapped = self._paths()
        clearance, _ = self.csdf_model(wrapped.reshape(-1, 6))
        clearance = clearance.reshape(len(wrapped), self.output_points)
        minimum = clearance.amin(dim=1)
        mean_clearance = clearance.mean(dim=1)
        violation_fraction = (clearance < self.margin_m).float().mean(dim=1)
        # Evaluate smoothness on a fixed reference grid so denser collision
        # sampling does not amplify float32 high-order difference noise.
        smoothness_path = continuous
        if self.derivative_basis is not self.basis:
            controls = self.original_controls.clone()
            controls[:, 1:-1] += self.residual
            smoothness_path = torch.einsum(
                "tc,bcj->btj", self.derivative_basis, controls
            )
            smoothness_path[:, 0] = self.original_controls[:, 0]
            smoothness_path[:, -1] = self.original_controls[:, -1]
        second = torch.diff(smoothness_path, n=2, dim=1)
        third = torch.diff(smoothness_path, n=3, dim=1)
        acceleration_proxy = torch.mean(second.square(), dim=(1, 2))
        jerk_proxy = torch.mean(third.square(), dim=(1, 2))
        deviation = torch.mean(self.residual.square(), dim=(1, 2))
        residual_difference = torch.diff(self.residual, dim=1)
        oscillation_proxy = torch.mean(residual_difference.square(), dim=(1, 2))
        acceleration_scale = max(
            float(self.proxy_scales.get("acceleration", 1.0)), 1.0e-12
        )
        jerk_scale = max(float(self.proxy_scales.get("jerk", 1.0)), 1.0e-12)
        oscillation_scale = max(
            float(self.proxy_scales.get("oscillation", 1.0)), 1.0e-12
        )
        potential = (
            float(self.weights["clearance"])
            * torch.clamp(minimum / self.margin_m, min=-10.0, max=2.0)
            - float(self.weights["collision_fraction"]) * violation_fraction
            - float(self.weights["deviation"]) * deviation
            - float(self.weights["acceleration"])
            * acceleration_proxy
            / acceleration_scale
            - float(self.weights["jerk"]) * jerk_proxy / jerk_scale
            - float(self.weights.get("oscillation", 0.0))
            * oscillation_proxy
            / oscillation_scale
        )
        metrics = {
            "min_clearance_m": minimum,
            "mean_clearance_m": mean_clearance,
            "violation_fraction": violation_fraction,
            "acceleration_proxy": acceleration_proxy,
            "jerk_proxy": jerk_proxy,
            "oscillation_proxy": oscillation_proxy,
            "deviation": deviation,
            "success": minimum >= self.margin_m,
            "potential": potential,
            "clearance_path": clearance,
        }
        self.last_metrics = metrics
        return wrapped, metrics

    def _state(self, wrapped: torch.Tensor, metrics: dict[str, torch.Tensor]) -> torch.Tensor:
        controls = wrapped[:, self.output_control_indices]
        original_wrapped = torch.remainder(self.original_controls + torch.pi, 2.0 * torch.pi) - torch.pi
        control_clearance = metrics["clearance_path"][:, self.output_control_indices]
        residual = self.residual / self.max_residual_rad
        step_fraction = torch.full(
            (len(wrapped), 1),
            self.current_step / max(self.episode_steps, 1),
            device=self.device,
        )
        summary = torch.stack(
            (
                metrics["min_clearance_m"] / self.margin_m,
                metrics["mean_clearance_m"] / self.margin_m,
                metrics["violation_fraction"],
            ),
            dim=1,
        )
        return torch.cat(
            (
                periodic_features(controls).flatten(1),
                periodic_features(original_wrapped).flatten(1),
                control_clearance / self.margin_m,
                residual.flatten(1),
                summary,
                step_fraction,
            ),
            dim=1,
        )

    def reset(
        self,
        batch_size: int | None = None,
        paths: np.ndarray | None = None,
        add_corruption: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if paths is None:
            if batch_size is None:
                raise ValueError("batch_size is required when paths are not supplied")
            indices = torch.randint(
                len(self.source_paths),
                (batch_size,),
                generator=self.generator,
                device=self.device,
            ).cpu().numpy()
            paths = self.source_paths[indices]
        self.original_controls = self._controls_from_paths(paths)
        batch_size = len(self.original_controls)
        self.residual = torch.zeros(
            (batch_size, self.control_count - 2, 6), device=self.device
        )
        if add_corruption and self.initial_corruption_rad > 0.0:
            noise = torch.randn(
                self.residual.shape, generator=self.generator, device=self.device
            )
            noise = F.avg_pool1d(
                noise.transpose(1, 2), kernel_size=5, stride=1, padding=2
            ).transpose(1, 2)
            scale = torch.rand(
                (batch_size, 1, 1), generator=self.generator, device=self.device
            )
            corruption = (noise * self.initial_corruption_rad * (0.35 + 0.65 * scale)).clamp(
                -self.max_residual_rad, self.max_residual_rad
            )
            # The policy sees only the corrupted input path; the safe source is
            # not retained as an observation or hidden target.
            self.original_controls[:, 1:-1] += corruption
        self.current_step = 0
        wrapped, metrics = self._measure()
        self.previous_potential = metrics["potential"]
        return self._state(wrapped, metrics), metrics

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if action.shape != (len(self.residual), self.action_dim):
            raise ValueError(f"expected action shape {(len(self.residual), self.action_dim)}")
        direction = self._safety_direction()
        delta = (
            action.clamp(-1.0, 1.0).unsqueeze(-1)
            * direction
            * self.action_step_rad
        )
        self.residual = (self.residual + delta).clamp(
            -self.max_residual_rad, self.max_residual_rad
        )
        self.current_step += 1
        wrapped, metrics = self._measure()
        reward = self.gamma * metrics["potential"] - self.previous_potential
        reward -= float(self.weights["action"]) * torch.mean(action.square(), dim=1)
        if action.shape[1] > 1:
            reward -= float(self.weights.get("action_oscillation", 0.0)) * torch.mean(
                torch.diff(action, dim=1).square(), dim=1
            )
        done = torch.full(
            (len(action),), self.current_step >= self.episode_steps, device=self.device, dtype=torch.bool
        )
        if done.all():
            reward += float(self.weights["success"]) * metrics["success"].float()
        self.previous_potential = metrics["potential"]
        return self._state(wrapped, metrics), reward, done, metrics

    def trajectories(self) -> np.ndarray:
        _, wrapped = self._paths()
        return wrapped.detach().cpu().numpy().astype(np.float32)
