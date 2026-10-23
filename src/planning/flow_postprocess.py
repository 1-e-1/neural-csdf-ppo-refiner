"""Goal-pose-only kinematic projection for Flow Matching inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.interpolate import CubicSpline

from .trajectory_math import JointTrajectory


@dataclass(frozen=True)
class TerminalIKResult:
    q: np.ndarray
    position_error_m: float
    orientation_error_rad: float
    objective: float
    restart_index: int


def _wrap_angles(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values) + np.pi) % (2.0 * np.pi) - np.pi


def retime_sampled_path(
    q: np.ndarray,
    predicted_duration: float,
    min_duration: float,
    nominal_max_duration: float,
    max_output_duration: float,
    velocity_limits: np.ndarray,
    acceleration_limits: np.ndarray,
    jerk_limits: np.ndarray,
    timing_points: int = 1024,
) -> tuple[JointTrajectory, dict[str, float | bool | int]]:
    """Densify a periodic path before computing an unclipped duration."""
    values = np.asarray(q, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6 or len(values) < 4:
        raise ValueError("q must have shape [T, 6] with at least four points")
    if timing_points < 4:
        raise ValueError("timing_points must be at least four")
    if not 0.0 < min_duration <= nominal_max_duration <= max_output_duration:
        raise ValueError("duration bounds must satisfy 0 < min <= nominal <= output")
    continuous = np.unwrap(_wrap_angles(values), axis=0, period=2.0 * np.pi)
    source_u = np.linspace(0.0, 1.0, len(continuous))
    timing_u = np.linspace(0.0, 1.0, timing_points)
    spline = CubicSpline(
        source_u,
        continuous,
        axis=0,
        bc_type=((1, np.zeros(6)), (1, np.zeros(6))),
    )
    dense_continuous = spline(timing_u)
    dense_continuous[0] = continuous[0]
    dense_continuous[-1] = continuous[-1]
    dq_du = spline(timing_u, 1)
    ddq_du2 = spline(timing_u, 2)
    dddq_du3 = spline(timing_u, 3)
    required_velocity = float(
        np.max(np.abs(dq_du) / np.asarray(velocity_limits, dtype=np.float64)[None])
    )
    required_acceleration = float(
        np.sqrt(
            np.max(
                np.abs(ddq_du2)
                / np.asarray(acceleration_limits, dtype=np.float64)[None]
            )
        )
    )
    required_jerk = float(
        np.cbrt(
            np.max(
                np.abs(dddq_du3)
                / np.asarray(jerk_limits, dtype=np.float64)[None]
            )
        )
    )
    required_duration = max(
        float(min_duration),
        required_velocity,
        required_acceleration,
        required_jerk,
    )
    duration = max(float(predicted_duration), required_duration)
    duration_valid = bool(np.isfinite(duration) and duration <= max_output_duration)
    trajectory = JointTrajectory(
        time=timing_u * duration,
        q=_wrap_angles(dense_continuous),
        dq=dq_du / duration,
        ddq=ddq_du2 / duration**2,
        dddq=dddq_du3 / duration**3,
        duration=duration,
        method="periodic_dense_unclipped_time_scaling",
    )
    trajectory.validate(timing_points)
    return trajectory, {
        "required_duration_s": required_duration,
        "predicted_duration_s": float(predicted_duration),
        "duration_s": duration,
        "duration_clipped_to_max": False,
        "duration_exceeds_nominal_max": duration > nominal_max_duration,
        "duration_valid": duration_valid,
        "nominal_max_duration_s": float(nominal_max_duration),
        "max_output_duration_s": float(max_output_duration),
        "timing_points": int(timing_points),
        "required_from_velocity_s": required_velocity,
        "required_from_acceleration_s": required_acceleration,
        "required_from_jerk_s": required_jerk,
    }


def refine_terminal_positions(
    fk_model,
    q_initial: np.ndarray,
    goal_position: np.ndarray,
    restarts: int,
    max_iterations: int,
    regularization_weight: float,
    seed: int,
    device: str,
) -> list[TerminalIKResult]:
    """Solve endpoint IK using position only, as required by the current task."""
    q_initial_tensor = torch.as_tensor(q_initial, dtype=torch.float32, device=device)
    goal_tensor = torch.as_tensor(goal_position, dtype=torch.float32, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    results = []
    for restart in range(restarts):
        initial = (
            q_initial_tensor.clone()
            if restart == 0
            else q_initial_tensor
            + 0.4 * restart * torch.randn(6, device=device, generator=generator)
        )
        q_parameter = initial.detach().clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [q_parameter],
            lr=0.8,
            max_iter=max_iterations,
            tolerance_grad=1.0e-9,
            tolerance_change=1.0e-11,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            position, _ = fk_model(q_parameter)
            loss = 100.0 * torch.sum((position - goal_tensor) ** 2)
            loss = loss + regularization_weight * torch.sum((q_parameter - initial) ** 2)
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            position, _ = fk_model(q_parameter)
            position_error = float(torch.linalg.norm(position - goal_tensor).cpu())
        results.append(
            TerminalIKResult(
                q=q_parameter.detach().cpu().numpy().astype(np.float64),
                position_error_m=position_error,
                orientation_error_rad=0.0,
                objective=position_error,
                restart_index=restart,
            )
        )
    return sorted(results, key=lambda result: result.position_error_m)


def refine_terminal_positions_batch(
    fk_model,
    q_initial: np.ndarray,
    goal_position: np.ndarray,
    max_iterations: int,
    regularization_weight: float,
    device: str,
) -> list[TerminalIKResult]:
    """Solve independent position-only endpoint IK problems in one batched LBFGS call."""
    q_initial_tensor = torch.as_tensor(q_initial, dtype=torch.float32, device=device)
    if q_initial_tensor.ndim != 2 or q_initial_tensor.shape[-1] != 6:
        raise ValueError("q_initial must have shape [candidates, 6]")
    goal_tensor = torch.as_tensor(goal_position, dtype=torch.float32, device=device)
    if goal_tensor.ndim == 1:
        goal_tensor = goal_tensor.unsqueeze(0).expand(q_initial_tensor.shape[0], -1)
    if goal_tensor.shape != (q_initial_tensor.shape[0], 3):
        raise ValueError("goal_position must have shape [3] or [candidates, 3]")
    q_parameter = q_initial_tensor.detach().clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [q_parameter],
        lr=0.8,
        max_iter=max_iterations,
        tolerance_grad=1.0e-9,
        tolerance_change=1.0e-11,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        position, _ = fk_model(q_parameter)
        position_loss = 100.0 * torch.sum((position - goal_tensor) ** 2, dim=-1)
        regularization = regularization_weight * torch.sum(
            (q_parameter - q_initial_tensor) ** 2,
            dim=-1,
        )
        loss = (position_loss + regularization).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        position, _ = fk_model(q_parameter)
        errors = torch.linalg.norm(position - goal_tensor, dim=-1).cpu().numpy()
        solutions = q_parameter.detach().cpu().numpy().astype(np.float64)
    return [
        TerminalIKResult(
            q=solutions[index],
            position_error_m=float(errors[index]),
            orientation_error_rad=0.0,
            objective=float(errors[index]),
            restart_index=0,
        )
        for index in range(solutions.shape[0])
    ]


def _orientation_error_rad(actual: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
    actual = actual / torch.linalg.norm(actual, dim=-1, keepdim=True).clamp_min(1.0e-12)
    desired = desired / torch.linalg.norm(desired, dim=-1, keepdim=True).clamp_min(1.0e-12)
    dot = torch.sum(actual * desired, dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def refine_terminal_configurations(
    fk_model,
    q_initial: np.ndarray,
    goal_position: np.ndarray,
    goal_quaternion_wxyz: np.ndarray,
    restarts: int,
    max_iterations: int,
    position_weight: float,
    orientation_weight: float,
    regularization_weight: float,
    seed: int,
    device: str,
) -> list[TerminalIKResult]:
    q_initial_tensor = torch.as_tensor(q_initial, dtype=torch.float32, device=device)
    goal_position_tensor = torch.as_tensor(goal_position, dtype=torch.float32, device=device)
    goal_quaternion_tensor = torch.as_tensor(goal_quaternion_wxyz, dtype=torch.float32, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    results = []
    for restart in range(restarts):
        if restart == 0:
            initial = q_initial_tensor.clone()
        else:
            initial = q_initial_tensor + 0.5 * restart * torch.randn(
                6, device=device, generator=generator
            )
        q_parameter = initial.detach().clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [q_parameter],
            lr=0.8,
            max_iter=max_iterations,
            tolerance_grad=1.0e-9,
            tolerance_change=1.0e-11,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            position, quaternion = fk_model(q_parameter)
            position_loss = torch.sum((position - goal_position_tensor) ** 2)
            quaternion_dot = torch.sum(quaternion * goal_quaternion_tensor).abs().clamp(max=1.0)
            orientation_loss = 1.0 - quaternion_dot**2
            regularization = torch.sum((q_parameter - initial) ** 2)
            loss = (
                position_weight * position_loss
                + orientation_weight * orientation_loss
                + regularization_weight * regularization
            )
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            position, quaternion = fk_model(q_parameter)
            position_error = torch.linalg.norm(position - goal_position_tensor)
            orientation_error = _orientation_error_rad(quaternion, goal_quaternion_tensor)
            objective = position_error + 0.1 * orientation_error
        results.append(
            TerminalIKResult(
                q=q_parameter.detach().cpu().numpy().astype(np.float64),
                position_error_m=float(position_error.cpu()),
                orientation_error_rad=float(orientation_error.cpu()),
                objective=float(objective.cpu()),
                restart_index=restart,
            )
        )
    return sorted(results, key=lambda result: result.objective)


def refine_terminal_configuration(
    fk_model,
    q_initial: np.ndarray,
    goal_position: np.ndarray,
    goal_quaternion_wxyz: np.ndarray,
    restarts: int,
    max_iterations: int,
    position_weight: float,
    orientation_weight: float,
    regularization_weight: float,
    seed: int,
    device: str,
) -> TerminalIKResult:
    """Return the best endpoint-only IK solution for backward compatibility."""
    return refine_terminal_configurations(
        fk_model=fk_model,
        q_initial=q_initial,
        goal_position=goal_position,
        goal_quaternion_wxyz=goal_quaternion_wxyz,
        restarts=restarts,
        max_iterations=max_iterations,
        position_weight=position_weight,
        orientation_weight=orientation_weight,
        regularization_weight=regularization_weight,
        seed=seed,
        device=device,
    )[0]


def smooth_and_time_parameterize(
    raw_q: np.ndarray,
    q_start: np.ndarray,
    q_terminal: np.ndarray,
    predicted_duration: float,
    points: int,
    control_point_count: int,
    residual_weight: float,
    min_duration: float,
    max_duration: float,
    velocity_limits: np.ndarray,
    acceleration_limits: np.ndarray,
    jerk_limits: np.ndarray,
) -> tuple[JointTrajectory, dict[str, float | bool]]:
    raw_q = np.asarray(raw_q, dtype=np.float64)
    control_indices = np.linspace(0, raw_q.shape[0] - 1, control_point_count).round().astype(int)
    controls = raw_q[control_indices].copy()
    knot_u = np.linspace(0.0, 1.0, control_point_count)
    raw_chord = raw_q[0][None, :] + knot_u[:, None] * (raw_q[-1] - raw_q[0])[None, :]
    raw_residual = controls - raw_chord
    minimum_jerk_phase = 10.0 * knot_u**3 - 15.0 * knot_u**4 + 6.0 * knot_u**5
    reference_controls = np.asarray(q_start)[None, :] + minimum_jerk_phase[:, None] * (
        np.asarray(q_terminal) - np.asarray(q_start)
    )[None, :]
    controls = reference_controls + float(residual_weight) * raw_residual
    controls[0] = np.asarray(q_start, dtype=np.float64)
    controls[-1] = np.asarray(q_terminal, dtype=np.float64)
    sample_u = np.linspace(0.0, 1.0, points)
    spline = CubicSpline(
        knot_u,
        controls,
        axis=0,
        bc_type=((1, np.zeros(6)), (1, np.zeros(6))),
    )
    q = spline(sample_u)
    q[0] = q_start
    q[-1] = q_terminal
    dq_du = spline(sample_u, 1)
    ddq_du2 = spline(sample_u, 2)
    dddq_du3 = spline(sample_u, 3)
    required_velocity = float(np.max(np.abs(dq_du) / np.asarray(velocity_limits)[None, :]))
    required_acceleration = float(
        np.sqrt(np.max(np.abs(ddq_du2) / np.asarray(acceleration_limits)[None, :]))
    )
    required_jerk = float(
        np.cbrt(np.max(np.abs(dddq_du3) / np.asarray(jerk_limits)[None, :]))
    )
    required_duration = max(min_duration, required_velocity, required_acceleration, required_jerk)
    duration = min(max(float(predicted_duration), required_duration), max_duration)
    trajectory = JointTrajectory(
        time=sample_u * duration,
        q=q,
        dq=dq_du / duration,
        ddq=ddq_du2 / duration**2,
        dddq=dddq_du3 / duration**3,
        duration=duration,
        method="flow_matching_ik_projected_cubic_spline",
    )
    trajectory.validate(points)
    return trajectory, {
        "required_duration_s": required_duration,
        "duration_clipped_to_max": required_duration > max_duration,
        "required_from_velocity_s": required_velocity,
        "required_from_acceleration_s": required_acceleration,
        "required_from_jerk_s": required_jerk,
    }


def geometry_preserving_terminal_projection(
    raw_q: np.ndarray,
    q_start: np.ndarray,
    q_terminal: np.ndarray,
    predicted_duration: float,
    min_duration: float,
    max_duration: float,
    velocity_limits: np.ndarray,
    acceleration_limits: np.ndarray,
    jerk_limits: np.ndarray,
    control_point_count: int = 16,
) -> tuple[JointTrajectory, dict[str, float | bool]]:
    """Inject endpoint IK with a minimum-jerk correction while retaining path geometry."""
    raw_q = np.asarray(raw_q, dtype=np.float64)
    if raw_q.ndim != 2 or raw_q.shape[1] != 6:
        raise ValueError("raw_q must have shape [points, 6]")
    q_start = np.asarray(q_start, dtype=np.float64)
    q_terminal = np.asarray(q_terminal, dtype=np.float64)
    sample_u = np.linspace(0.0, 1.0, len(raw_q))
    phase = 10.0 * sample_u**3 - 15.0 * sample_u**4 + 6.0 * sample_u**5
    start_offset = q_start - raw_q[0]
    terminal_offset = q_terminal - raw_q[-1]
    q = (
        raw_q
        + (1.0 - phase)[:, None] * start_offset[None]
        + phase[:, None] * terminal_offset[None]
    )
    q[0] = q_start
    q[-1] = q_terminal
    if not 4 <= control_point_count <= len(raw_q):
        raise ValueError("control_point_count must be in [4, len(raw_q)]")
    control_indices = np.linspace(0, len(raw_q) - 1, control_point_count).round().astype(int)
    knot_u = sample_u[control_indices]
    controls = q[control_indices]
    spline = CubicSpline(
        knot_u,
        controls,
        axis=0,
        bc_type=((1, np.zeros(6)), (1, np.zeros(6))),
    )
    q = spline(sample_u)
    q[0] = q_start
    q[-1] = q_terminal
    dq_du = spline(sample_u, 1)
    ddq_du2 = spline(sample_u, 2)
    dddq_du3 = spline(sample_u, 3)
    required_velocity = float(np.max(np.abs(dq_du) / np.asarray(velocity_limits)[None]))
    required_acceleration = float(
        np.sqrt(np.max(np.abs(ddq_du2) / np.asarray(acceleration_limits)[None]))
    )
    required_jerk = float(
        np.cbrt(np.max(np.abs(dddq_du3) / np.asarray(jerk_limits)[None]))
    )
    required_duration = max(min_duration, required_velocity, required_acceleration, required_jerk)
    duration = min(max(float(predicted_duration), required_duration), max_duration)
    trajectory = JointTrajectory(
        time=sample_u * duration,
        q=q,
        dq=dq_du / duration,
        ddq=ddq_du2 / duration**2,
        dddq=dddq_du3 / duration**3,
        duration=duration,
        method="flow_matching_geometry_preserving_terminal_projection",
    )
    trajectory.validate(len(raw_q))
    return trajectory, {
        "required_duration_s": required_duration,
        "duration_clipped_to_max": required_duration > max_duration,
        "required_from_velocity_s": required_velocity,
        "required_from_acceleration_s": required_acceleration,
        "required_from_jerk_s": required_jerk,
    }
