"""Periodic trajectory utilities and exact capsule self-collision queries."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.interpolate import CubicSpline
import torch
from torch.nn import functional as F

from h1_1111_grasp.trajectory_safety import link_segments, self_clearance_batch


TAU = 2.0 * np.pi


def wrap_angles(values: np.ndarray) -> np.ndarray:
    """Map periodic joint values to the half-open interval [-pi, pi)."""
    values = np.asarray(values)
    return (values + np.pi) % TAU - np.pi


def shortest_angular_delta(end: np.ndarray, start: np.ndarray) -> np.ndarray:
    return wrap_angles(np.asarray(end) - np.asarray(start))


def unwrap_trajectory(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 6:
        raise ValueError("q must have shape [T, 6]")
    unwrapped = np.unwrap(wrap_angles(q), axis=0, period=TAU)
    unwrapped += shortest_angular_delta(q[0], unwrapped[0])[None]
    return unwrapped


def periodic_resample(q: np.ndarray, output_points: int = 256) -> np.ndarray:
    """Cubic periodic-aware resampling with fixed endpoints."""
    if output_points < 2:
        raise ValueError("output_points must be at least two")
    continuous = unwrap_trajectory(q)
    source_u = np.linspace(0.0, 1.0, len(continuous))
    target_u = np.linspace(0.0, 1.0, output_points)
    spline = CubicSpline(
        source_u,
        continuous,
        axis=0,
        bc_type=((1, np.zeros(6)), (1, np.zeros(6))),
    )
    output = spline(target_u)
    output[0] = continuous[0]
    output[-1] = continuous[-1]
    return wrap_angles(output).astype(np.float32)


def cubic_interpolation_matrix(control_points: int, output_points: int) -> np.ndarray:
    """Return a fixed cubic-spline basis matrix [output_points, control_points]."""
    if control_points < 4:
        raise ValueError("control_points must be at least four")
    controls_u = np.linspace(0.0, 1.0, control_points)
    output_u = np.linspace(0.0, 1.0, output_points)
    basis = np.empty((output_points, control_points), dtype=np.float32)
    identity = np.eye(control_points, dtype=np.float64)
    for index in range(control_points):
        spline = CubicSpline(
            controls_u,
            identity[:, index],
            bc_type=((1, 0.0), (1, 0.0)),
        )
        basis[:, index] = spline(output_u).astype(np.float32)
    basis[0] = 0.0
    basis[0, 0] = 1.0
    basis[-1] = 0.0
    basis[-1, -1] = 1.0
    return basis


class ExactSelfCollisionOracle:
    """Capsule-based collision labeler backed by the existing exact segment query."""

    def __init__(
        self,
        fk_model: torch.nn.Module,
        capsule_radius_m: float = 0.045,
        adjacency: int = 2,
        device: str = "cpu",
        batch_size: int = 8192,
    ) -> None:
        self.fk_model = fk_model
        self.capsule_radius_m = float(capsule_radius_m)
        self.adjacency = int(adjacency)
        self.device = device
        self.batch_size = int(batch_size)

    def clearance(self, q: np.ndarray) -> np.ndarray:
        values = wrap_angles(np.asarray(q, dtype=np.float32)).reshape(-1, 6)
        outputs = []
        for start in range(0, len(values), self.batch_size):
            batch = values[start : start + self.batch_size]
            segments = link_segments(self.fk_model, batch, device=self.device)
            outputs.append(
                self_clearance_batch(
                    segments,
                    capsule_radius_m=self.capsule_radius_m,
                    adjacency=self.adjacency,
                ).astype(np.float32)
            )
        return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)


@dataclass(frozen=True)
class ContinuousSafetyReport:
    safe: bool
    min_clearance_m: float
    checked_configurations: int
    source_points: int
    max_interval_step_rad: float
    required_clearance_m: float

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


@dataclass(frozen=True)
class SafeTerminalResult:
    q: np.ndarray
    found: bool
    position_error_m: float
    predicted_clearance_m: float
    exact_clearance_m: float
    restart_index: int


def refine_safe_terminal_configuration(
    fk_model: torch.nn.Module,
    csdf_model: torch.nn.Module,
    oracle: ExactSelfCollisionOracle,
    q_initial: np.ndarray,
    required_clearance_m: float,
    device: str,
    restarts: int = 32,
    iterations: int = 300,
    position_tolerance_m: float = 0.003,
    seed: int = 20260802,
) -> SafeTerminalResult:
    """Find a safer joint solution while preserving the Cartesian TCP target."""
    q_initial_tensor = torch.as_tensor(q_initial, dtype=torch.float32, device=device)
    with torch.no_grad():
        target_position, _ = fk_model(q_initial_tensor)
    generator = torch.Generator(device=device).manual_seed(seed)
    initial = q_initial_tensor[None].repeat(restarts, 1)
    if restarts > 1:
        initial[1:] += 0.5 * torch.randn(
            (restarts - 1, 6), generator=generator, device=device
        )
    parameter = torch.nn.Parameter(initial)
    optimizer = torch.optim.Adam([parameter], lr=0.03)
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        wrapped = torch.remainder(parameter + torch.pi, 2.0 * torch.pi) - torch.pi
        position, _ = fk_model(wrapped)
        predicted_clearance, _ = csdf_model(wrapped)
        position_loss = torch.sum((position - target_position[None]) ** 2, dim=1)
        safety = F.softplus(
            (required_clearance_m + 0.004 - predicted_clearance) / 0.003
        ) * 0.003
        regularization = torch.mean(
            (torch.remainder(wrapped - q_initial_tensor + torch.pi, 2.0 * torch.pi) - torch.pi) ** 2,
            dim=1,
        )
        loss = torch.mean(1000.0 * position_loss + 5000.0 * safety.square() + 0.002 * regularization)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            parameter.copy_(torch.remainder(parameter + torch.pi, 2.0 * torch.pi) - torch.pi)
    with torch.no_grad():
        wrapped = torch.remainder(parameter + torch.pi, 2.0 * torch.pi) - torch.pi
        positions, _ = fk_model(wrapped)
        position_errors = torch.linalg.norm(positions - target_position[None], dim=1).cpu().numpy()
        predicted, _ = csdf_model(wrapped)
        predicted = predicted.cpu().numpy()
        candidates = wrapped.cpu().numpy()
    exact = oracle.clearance(candidates)
    valid = (position_errors <= position_tolerance_m) & (exact >= required_clearance_m)
    score = 100.0 * position_errors + 1000.0 * np.maximum(0.0, required_clearance_m - exact)
    if valid.any():
        valid_indices = np.flatnonzero(valid)
        selected = int(valid_indices[np.argmin(score[valid_indices])])
    else:
        selected = int(np.argmin(score))
    return SafeTerminalResult(
        q=candidates[selected].astype(np.float32),
        found=bool(valid[selected]),
        position_error_m=float(position_errors[selected]),
        predicted_clearance_m=float(predicted[selected]),
        exact_clearance_m=float(exact[selected]),
        restart_index=selected,
    )


def refine_safe_terminal_configurations_batch(
    fk_model: torch.nn.Module,
    csdf_model: torch.nn.Module,
    oracle: ExactSelfCollisionOracle,
    q_initial: np.ndarray,
    required_clearance_m: float,
    device: str,
    target_positions: np.ndarray | None = None,
    restarts: int = 32,
    iterations: int = 300,
    position_tolerance_m: float = 0.003,
    seed: int = 20260802,
) -> list[SafeTerminalResult]:
    """Batch the safe terminal search for independent Cartesian targets."""
    q_initial_array = np.asarray(q_initial, dtype=np.float32)
    if q_initial_array.ndim != 2 or q_initial_array.shape[1] != 6:
        raise ValueError("q_initial must have shape [B, 6]")
    if len(q_initial_array) == 0:
        return []
    q_initial_tensor = torch.as_tensor(
        q_initial_array, dtype=torch.float32, device=device
    )
    if target_positions is None:
        with torch.no_grad():
            target_tensor, _ = fk_model(q_initial_tensor)
    else:
        target_array = np.asarray(target_positions, dtype=np.float32)
        if target_array.shape != (len(q_initial_array), 3):
            raise ValueError("target_positions must have shape [B, 3]")
        target_tensor = torch.as_tensor(
            target_array, dtype=torch.float32, device=device
        )

    generator = torch.Generator(device=device).manual_seed(seed)
    initial = q_initial_tensor[:, None, :].repeat(1, restarts, 1)
    if restarts > 1:
        initial[:, 1:] += 0.5 * torch.randn(
            (len(initial), restarts - 1, 6),
            generator=generator,
            device=device,
        )
    parameter = torch.nn.Parameter(initial)
    optimizer = torch.optim.Adam([parameter], lr=0.03)
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        wrapped = torch.remainder(parameter + torch.pi, 2.0 * torch.pi) - torch.pi
        flat = wrapped.reshape(-1, 6)
        position, _ = fk_model(flat)
        position = position.reshape(len(initial), restarts, 3)
        predicted_clearance, _ = csdf_model(flat)
        predicted_clearance = predicted_clearance.reshape(len(initial), restarts)
        position_loss = torch.sum(
            (position - target_tensor[:, None, :]) ** 2, dim=2
        )
        safety = F.softplus(
            (required_clearance_m + 0.004 - predicted_clearance) / 0.003
        ) * 0.003
        angular_delta = torch.remainder(
            wrapped - q_initial_tensor[:, None, :] + torch.pi,
            2.0 * torch.pi,
        ) - torch.pi
        regularization = torch.mean(angular_delta.square(), dim=2)
        loss = torch.mean(
            1000.0 * position_loss
            + 5000.0 * safety.square()
            + 0.002 * regularization
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            parameter.copy_(
                torch.remainder(parameter + torch.pi, 2.0 * torch.pi) - torch.pi
            )

    with torch.no_grad():
        wrapped = torch.remainder(parameter + torch.pi, 2.0 * torch.pi) - torch.pi
        flat = wrapped.reshape(-1, 6)
        positions, _ = fk_model(flat)
        positions = positions.reshape(len(initial), restarts, 3)
        position_errors = torch.linalg.norm(
            positions - target_tensor[:, None, :], dim=2
        ).cpu().numpy()
        predicted, _ = csdf_model(flat)
        predicted = predicted.reshape(len(initial), restarts).cpu().numpy()
        candidates = wrapped.cpu().numpy()
    exact = oracle.clearance(candidates.reshape(-1, 6)).reshape(
        len(initial), restarts
    )
    results = []
    for batch_index in range(len(initial)):
        valid = (
            (position_errors[batch_index] <= position_tolerance_m)
            & (exact[batch_index] >= required_clearance_m)
        )
        score = (
            100.0 * position_errors[batch_index]
            + 1000.0
            * np.maximum(0.0, required_clearance_m - exact[batch_index])
        )
        if valid.any():
            valid_indices = np.flatnonzero(valid)
            selected = int(valid_indices[np.argmin(score[valid_indices])])
        else:
            selected = int(np.argmin(score))
        results.append(
            SafeTerminalResult(
                q=candidates[batch_index, selected].astype(np.float32),
                found=bool(valid[selected]),
                position_error_m=float(position_errors[batch_index, selected]),
                predicted_clearance_m=float(predicted[batch_index, selected]),
                exact_clearance_m=float(exact[batch_index, selected]),
                restart_index=selected,
            )
        )
    return results


def project_terminal_with_minimum_jerk(
    q: np.ndarray, q_terminal: np.ndarray
) -> np.ndarray:
    """Spread a terminal joint correction over the path with zero endpoint rates."""
    continuous = unwrap_trajectory(q)
    delta = shortest_angular_delta(q_terminal, q[-1])
    u = np.linspace(0.0, 1.0, len(q))
    phase = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    corrected = continuous + phase[:, None] * delta[None]
    corrected[0] = continuous[0]
    corrected[-1] = continuous[-1] + delta
    return wrap_angles(corrected).astype(np.float32)


def adaptive_validate_trajectory(
    q: np.ndarray,
    oracle: ExactSelfCollisionOracle,
    required_clearance_m: float = 0.010,
    max_joint_step_rad: float = 0.025,
    near_boundary_extra_subdivisions: int = 2,
) -> ContinuousSafetyReport:
    """Densely check every interval, refining again near the collision boundary.

    This is an adaptive discrete approximation to continuous collision checking.
    It does not claim an analytic swept-volume guarantee.
    """
    continuous = unwrap_trajectory(q)
    dense = [continuous[0]]
    for left, right in zip(continuous[:-1], continuous[1:]):
        steps = max(1, int(np.ceil(np.max(np.abs(right - left)) / max_joint_step_rad)))
        for index in range(1, steps + 1):
            dense.append(left + (right - left) * (index / steps))
    dense_array = np.asarray(dense, dtype=np.float32)
    clearance = oracle.clearance(dense_array)

    for _ in range(max(0, near_boundary_extra_subdivisions)):
        near = np.flatnonzero(clearance[:-1] < required_clearance_m + 0.015)
        if len(near) == 0:
            break
        midpoints = 0.5 * (dense_array[near] + dense_array[near + 1])
        dense_array = np.concatenate((dense_array, midpoints), axis=0)
        clearance = oracle.clearance(dense_array)

    minimum = float(clearance.min())
    return ContinuousSafetyReport(
        safe=minimum >= required_clearance_m,
        min_clearance_m=minimum,
        checked_configurations=len(dense_array),
        source_points=len(q),
        max_interval_step_rad=float(max_joint_step_rad),
        required_clearance_m=float(required_clearance_m),
    )
