"""Deterministic joint-trajectory baselines and derivatives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointTrajectory:
    time: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray
    dddq: np.ndarray
    duration: float
    method: str

    def validate(self, points: int, dof: int = 6) -> None:
        expected = (points, dof)
        for name in ("q", "dq", "ddq", "dddq"):
            value = getattr(self, name)
            if value.shape != expected or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite with shape {expected}, got {value.shape}")
        if self.time.shape != (points,) or not np.all(np.diff(self.time) > 0.0):
            raise ValueError("time must be finite, one-dimensional, and strictly increasing")
        if not np.isclose(self.time[0], 0.0) or not np.isclose(self.time[-1], self.duration):
            raise ValueError("time endpoints must be [0, duration]")


def _as_joint_vector(value: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if out.shape != (6,) or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite with shape (6,), got {out.shape}")
    return out


def minimum_jerk_duration(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    velocity_limits: np.ndarray,
    acceleration_limits: np.ndarray,
    jerk_limits: np.ndarray,
    min_duration: float,
    max_duration: float,
) -> tuple[float, bool, dict[str, float]]:
    """Choose duration from exact peak factors of a quintic minimum-jerk curve."""
    delta = np.abs(_as_joint_vector(q_goal, "q_goal") - _as_joint_vector(q_start, "q_start"))
    velocity_limits = _as_joint_vector(velocity_limits, "velocity_limits")
    acceleration_limits = _as_joint_vector(acceleration_limits, "acceleration_limits")
    jerk_limits = _as_joint_vector(jerk_limits, "jerk_limits")
    if np.any(velocity_limits <= 0) or np.any(acceleration_limits <= 0) or np.any(jerk_limits <= 0):
        raise ValueError("Planning soft limits must be positive")

    required_v = float(np.max(1.875 * delta / velocity_limits))
    required_a = float(np.max(np.sqrt((10.0 / np.sqrt(3.0)) * delta / acceleration_limits)))
    required_j = float(np.max(np.cbrt(60.0 * delta / jerk_limits)))
    required = max(float(min_duration), required_v, required_a, required_j)
    duration = min(required, float(max_duration))
    return duration, required <= max_duration + 1.0e-12, {
        "required_from_velocity_s": required_v,
        "required_from_acceleration_s": required_a,
        "required_from_jerk_s": required_j,
        "unclipped_required_duration_s": required,
    }


def minimum_jerk_trajectory(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    duration: float,
    points: int = 64,
) -> JointTrajectory:
    q_start = _as_joint_vector(q_start, "q_start")
    q_goal = _as_joint_vector(q_goal, "q_goal")
    if duration <= 0.0 or points < 4:
        raise ValueError("duration must be positive and points must be at least 4")
    time = np.linspace(0.0, duration, points, dtype=np.float64)
    u = time / duration
    delta = q_goal - q_start

    s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    ds = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / duration
    dds = (60.0 * u - 180.0 * u**2 + 120.0 * u**3) / duration**2
    ddds = (60.0 - 360.0 * u + 360.0 * u**2) / duration**3

    result = JointTrajectory(
        time=time,
        q=q_start[None, :] + s[:, None] * delta[None, :],
        dq=ds[:, None] * delta[None, :],
        ddq=dds[:, None] * delta[None, :],
        dddq=ddds[:, None] * delta[None, :],
        duration=float(duration),
        method="minimum_jerk",
    )
    result.validate(points)
    return result


def linear_trajectory(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    duration: float,
    points: int = 64,
) -> JointTrajectory:
    q_start = _as_joint_vector(q_start, "q_start")
    q_goal = _as_joint_vector(q_goal, "q_goal")
    if duration <= 0.0 or points < 2:
        raise ValueError("duration must be positive and points must be at least 2")
    time = np.linspace(0.0, duration, points, dtype=np.float64)
    u = time / duration
    delta = q_goal - q_start
    result = JointTrajectory(
        time=time,
        q=q_start[None, :] + u[:, None] * delta[None, :],
        dq=np.broadcast_to(delta[None, :] / duration, (points, 6)).copy(),
        ddq=np.zeros((points, 6), dtype=np.float64),
        dddq=np.zeros((points, 6), dtype=np.float64),
        duration=float(duration),
        method="linear",
    )
    result.validate(points)
    return result


def trajectory_from_positions(q: np.ndarray, duration: float) -> JointTrajectory:
    """Create timestamped derivatives for an arbitrary fixed-phase trajectory."""
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 6 or q.shape[0] < 4 or not np.all(np.isfinite(q)):
        raise ValueError("q must be finite with shape (N>=4, 6)")
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    time = np.linspace(0.0, duration, q.shape[0], dtype=np.float64)
    dq = np.gradient(q, time, axis=0, edge_order=2)
    ddq = np.gradient(dq, time, axis=0, edge_order=2)
    dddq = np.gradient(ddq, time, axis=0, edge_order=2)
    result = JointTrajectory(time, q.copy(), dq, ddq, dddq, float(duration), "sampled_positions")
    result.validate(q.shape[0])
    return result
