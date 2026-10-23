"""Visual point-cloud safety screening for 64x6 arm trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class CandidateMetrics:
    index: int
    valid: bool
    score: float
    terminal_error_m: float
    environment_clearance_m: float
    self_clearance_m: float
    velocity_rms_rad_s: float
    acceleration_rms_rad_s2: float
    jerk_rms_rad_s3: float

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


def _point_segment_distance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    direction = end - start
    denominator = float(direction @ direction)
    if denominator < 1.0e-12:
        return np.linalg.norm(points - start, axis=-1)
    fraction = np.clip(((points - start) @ direction) / denominator, 0.0, 1.0)
    closest = start + fraction[:, None] * direction
    return np.linalg.norm(points - closest, axis=-1)


def _segment_distance(p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray) -> float:
    """Shortest distance between two finite 3-D segments."""
    u = p1 - p0
    v = q1 - q0
    w = p0 - q0
    a = float(u @ u)
    b = float(u @ v)
    c = float(v @ v)
    d = float(u @ w)
    e = float(v @ w)
    denominator = a * c - b * b
    if a < 1.0e-12 and c < 1.0e-12:
        return float(np.linalg.norm(p0 - q0))
    if a < 1.0e-12:
        return float(_point_segment_distance(p0[None], q0, q1)[0])
    if c < 1.0e-12:
        return float(_point_segment_distance(q0[None], p0, p1)[0])
    s_numerator = 0.0 if denominator < 1.0e-12 else b * e - c * d
    s_denominator = 1.0 if denominator < 1.0e-12 else denominator
    t_numerator = a * e - b * d
    t_denominator = denominator
    if s_numerator < 0.0:
        s_numerator = 0.0
        t_numerator, t_denominator = e, c
    elif s_numerator > s_denominator:
        s_numerator = s_denominator
        t_numerator, t_denominator = e + b, c
    if t_numerator < 0.0:
        t_numerator = 0.0
        if -d < 0.0:
            s_numerator = 0.0
        elif -d > a:
            s_numerator, s_denominator = s_denominator, s_denominator
        else:
            s_numerator, s_denominator = -d, a
    elif t_numerator > t_denominator:
        t_numerator = t_denominator
        if -d + b < 0.0:
            s_numerator = 0.0
        elif -d + b > a:
            s_numerator, s_denominator = s_denominator, s_denominator
        else:
            s_numerator, s_denominator = -d + b, a
    sc = 0.0 if abs(s_numerator) < 1.0e-12 else s_numerator / s_denominator
    tc = 0.0 if abs(t_numerator) < 1.0e-12 else t_numerator / t_denominator
    return float(np.linalg.norm(w + sc * u - tc * v))


def _segment_distance_batch(
    p0: np.ndarray,
    p1: np.ndarray,
    q0: np.ndarray,
    q1: np.ndarray,
) -> np.ndarray:
    """Vectorized counterpart of :func:`_segment_distance`."""
    u = p1 - p0
    v = q1 - q0
    w = p0 - q0
    a = np.einsum("ij,ij->i", u, u)
    b = np.einsum("ij,ij->i", u, v)
    c = np.einsum("ij,ij->i", v, v)
    d = np.einsum("ij,ij->i", u, w)
    e = np.einsum("ij,ij->i", v, w)
    denominator = a * c - b * b
    parallel = denominator < 1.0e-12
    s_numerator = np.where(parallel, 0.0, b * e - c * d)
    s_denominator = np.where(parallel, 1.0, denominator)
    t_numerator = a * e - b * d
    t_denominator = denominator.copy()

    before_start = s_numerator < 0.0
    s_numerator = np.where(before_start, 0.0, s_numerator)
    t_numerator = np.where(before_start, e, t_numerator)
    t_denominator = np.where(before_start, c, t_denominator)
    after_end = s_numerator > s_denominator
    s_numerator = np.where(after_end, s_denominator, s_numerator)
    t_numerator = np.where(after_end, e + b, t_numerator)
    t_denominator = np.where(after_end, c, t_denominator)

    before_q = t_numerator < 0.0
    t_numerator = np.where(before_q, 0.0, t_numerator)
    s_before = np.where(-d < 0.0, 0.0, np.where(-d > a, s_denominator, -d))
    s_den_before = np.where((-d >= 0.0) & (-d <= a), a, s_denominator)
    s_numerator = np.where(before_q, s_before, s_numerator)
    s_denominator = np.where(before_q, s_den_before, s_denominator)

    after_q = t_numerator > t_denominator
    t_numerator = np.where(after_q, t_denominator, t_numerator)
    shifted = -d + b
    s_after = np.where(shifted < 0.0, 0.0, np.where(shifted > a, s_denominator, shifted))
    s_den_after = np.where((shifted >= 0.0) & (shifted <= a), a, s_denominator)
    s_numerator = np.where(after_q, s_after, s_numerator)
    s_denominator = np.where(after_q, s_den_after, s_denominator)

    sc = np.divide(
        s_numerator,
        s_denominator,
        out=np.zeros_like(s_numerator),
        where=np.abs(s_denominator) >= 1.0e-12,
    )
    tc = np.divide(
        t_numerator,
        t_denominator,
        out=np.zeros_like(t_numerator),
        where=np.abs(t_denominator) >= 1.0e-12,
    )
    return np.linalg.norm(w + sc[:, None] * u - tc[:, None] * v, axis=1)


def link_segments(fk_model: torch.nn.Module, q_path: np.ndarray, device: str = "cpu") -> np.ndarray:
    q = torch.as_tensor(q_path, dtype=torch.float32, device=device)
    with torch.no_grad():
        positions, _ = fk_model.link_poses(q)
    positions_np = positions.detach().cpu().numpy().astype(np.float64)
    base = np.zeros((*positions_np.shape[:-2], 1, 3), dtype=np.float64)
    points = np.concatenate((base, positions_np), axis=-2)
    return np.stack((points[..., :-1, :], points[..., 1:, :]), axis=-2)


def environment_clearance(
    segments: np.ndarray,
    scene_points: np.ndarray,
    capsule_radius_m: float,
) -> float:
    cloud = np.asarray(scene_points, dtype=np.float64).reshape(-1, 3)
    if len(cloud) == 0:
        return float("inf")
    minimum = float("inf")
    for start, end in np.asarray(segments).reshape(-1, 2, 3):
        minimum = min(minimum, float(_point_segment_distance(cloud, start, end).min()))
    return minimum - float(capsule_radius_m)


def environment_clearance_batch(
    candidate_segments: np.ndarray,
    scene_points: np.ndarray,
    capsule_radius_m: float,
    segment_chunk_size: int = 64,
) -> np.ndarray:
    """Return point-cloud clearance for each candidate using bounded-memory chunks."""
    segments = np.asarray(candidate_segments, dtype=np.float64)
    if segments.ndim < 4 or segments.shape[-2:] != (2, 3):
        raise ValueError("candidate_segments must have shape [K, ..., 2, 3]")
    cloud = np.asarray(scene_points, dtype=np.float64).reshape(-1, 3)
    if len(cloud) == 0:
        return np.full(len(segments), np.inf, dtype=np.float64)
    candidate_count = len(segments)
    flat = segments.reshape(candidate_count, -1, 2, 3)
    minima = np.full((candidate_count, flat.shape[1]), np.inf, dtype=np.float64)
    all_segments = flat.reshape(-1, 2, 3)
    all_minima = minima.reshape(-1)
    for start_index in range(0, len(all_segments), segment_chunk_size):
        batch = all_segments[start_index : start_index + segment_chunk_size]
        start = batch[:, 0]
        direction = batch[:, 1] - start
        denominator = np.einsum("ij,ij->i", direction, direction)
        relative = cloud[None, :, :] - start[:, None, :]
        fraction = np.divide(
            np.einsum("spj,sj->sp", relative, direction),
            denominator[:, None],
            out=np.zeros((len(batch), len(cloud)), dtype=np.float64),
            where=denominator[:, None] >= 1.0e-12,
        )
        fraction = np.clip(fraction, 0.0, 1.0)
        closest_delta = relative - fraction[:, :, None] * direction[:, None, :]
        all_minima[start_index : start_index + len(batch)] = np.sqrt(
            np.einsum("spj,spj->sp", closest_delta, closest_delta)
        ).min(axis=1)
    return minima.min(axis=1) - float(capsule_radius_m)


def self_clearance(
    segments: np.ndarray,
    capsule_radius_m: float,
    adjacency: int = 2,
) -> float:
    minimum = float("inf")
    for frame in np.asarray(segments):
        for left in range(len(frame)):
            if np.linalg.norm(frame[left, 1] - frame[left, 0]) < 1.0e-8:
                continue
            for right in range(left + adjacency + 1, len(frame)):
                if np.linalg.norm(frame[right, 1] - frame[right, 0]) < 1.0e-8:
                    continue
                minimum = min(minimum, _segment_distance(*frame[left], *frame[right]))
    return minimum - 2.0 * float(capsule_radius_m)


def self_clearance_batch(
    segments: np.ndarray,
    capsule_radius_m: float,
    adjacency: int = 2,
) -> np.ndarray:
    """Return self-clearance for every frame in a segment batch."""
    frames = np.asarray(segments, dtype=np.float64)
    if frames.ndim != 4 or frames.shape[-2:] != (2, 3):
        raise ValueError("segments must have shape [N, L, 2, 3]")
    minimum = np.full(len(frames), np.inf, dtype=np.float64)
    lengths = np.linalg.norm(frames[:, :, 1] - frames[:, :, 0], axis=-1)
    for left in range(frames.shape[1]):
        for right in range(left + adjacency + 1, frames.shape[1]):
            valid = (lengths[:, left] >= 1.0e-8) & (lengths[:, right] >= 1.0e-8)
            if not valid.any():
                continue
            distances = _segment_distance_batch(
                frames[:, left, 0],
                frames[:, left, 1],
                frames[:, right, 0],
                frames[:, right, 1],
            )
            minimum[valid] = np.minimum(minimum[valid], distances[valid])
    return minimum - 2.0 * float(capsule_radius_m)


def trajectory_derivatives(q_path: np.ndarray, duration_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray(q_path, dtype=np.float64)
    if q.shape != (64, 6):
        raise ValueError(f"expected trajectory shape (64, 6), got {q.shape}")
    if not np.isfinite(q).all() or duration_s <= 0.0:
        raise ValueError("trajectory must be finite and duration must be positive")
    time = np.linspace(0.0, float(duration_s), 64)
    continuous_q = np.unwrap(q, axis=0, period=2.0 * np.pi)
    velocity = np.gradient(continuous_q, time, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, time, axis=0, edge_order=2)
    jerk = np.gradient(acceleration, time, axis=0, edge_order=2)
    return velocity, acceleration, jerk


def screen_candidates(
    fk_model: torch.nn.Module,
    candidates: np.ndarray,
    durations_s: np.ndarray,
    goal_position_m: np.ndarray,
    scene_points_m: np.ndarray,
    device: str = "cpu",
    capsule_radius_m: float = 0.055,
    required_clearance_m: float = 0.025,
) -> tuple[int | None, list[CandidateMetrics]]:
    paths = np.asarray(candidates, dtype=np.float64)
    durations = np.asarray(durations_s, dtype=np.float64).reshape(-1)
    if paths.ndim != 3 or paths.shape[1:] != (64, 6) or len(paths) != len(durations):
        raise ValueError("candidates must have shape [K, 64, 6] and one duration per candidate")
    goal = np.asarray(goal_position_m, dtype=np.float64).reshape(3)
    reports = []
    for index, (path, duration) in enumerate(zip(paths, durations)):
        segments = link_segments(fk_model, path, device=device)
        environment = environment_clearance(segments, scene_points_m, capsule_radius_m)
        self_distance = self_clearance(segments, capsule_radius_m)
        with torch.no_grad():
            terminal, _ = fk_model(torch.as_tensor(path[-1], dtype=torch.float32, device=device))
        terminal_error = float(np.linalg.norm(terminal.cpu().numpy() - goal))
        velocity, acceleration, jerk = trajectory_derivatives(path, float(duration))
        velocity_rms = float(np.sqrt(np.mean(velocity**2)))
        acceleration_rms = float(np.sqrt(np.mean(acceleration**2)))
        jerk_rms = float(np.sqrt(np.mean(jerk**2)))
        valid = environment >= required_clearance_m and self_distance >= required_clearance_m
        clearance_penalty = max(0.0, required_clearance_m - min(environment, self_distance))
        score = (
            10.0 * terminal_error
            + 40.0 * clearance_penalty
            + 0.01 * velocity_rms
            + 0.002 * acceleration_rms
            + 0.0002 * jerk_rms
        )
        reports.append(
            CandidateMetrics(
                index=index,
                valid=valid,
                score=score,
                terminal_error_m=terminal_error,
                environment_clearance_m=environment,
                self_clearance_m=self_distance,
                velocity_rms_rad_s=velocity_rms,
                acceleration_rms_rad_s2=acceleration_rms,
                jerk_rms_rad_s3=jerk_rms,
            )
        )
    valid_reports = [report for report in reports if report.valid]
    selected = min(valid_reports, key=lambda report: report.score).index if valid_reports else None
    return selected, reports
