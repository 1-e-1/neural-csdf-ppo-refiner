"""Six-dimensional periodic C-space sampling and C-SDF dataset generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from .geometry import ExactSelfCollisionOracle, wrap_angles


@dataclass(frozen=True)
class DatasetSummary:
    samples: int
    collision_samples: int
    collision_ratio: float
    uniform_collision_ratio: float
    boundary_samples: int
    clearance_min_m: float
    clearance_max_m: float
    seed: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def sample_periodic_configurations(count: int, seed: int) -> np.ndarray:
    """Combine scrambled Sobol points and independent uniform draws."""
    if count < 1:
        raise ValueError("count must be positive")
    sobol_count = count // 2
    engine = torch.quasirandom.SobolEngine(6, scramble=True, seed=seed)
    sobol = engine.draw(sobol_count).numpy()
    rng = np.random.default_rng(seed + 1)
    uniform = rng.random((count - sobol_count, 6))
    unit = np.concatenate((sobol, uniform), axis=0)
    rng.shuffle(unit)
    return (unit * (2.0 * np.pi) - np.pi).astype(np.float32)


def generate_csdf_dataset(
    oracle: ExactSelfCollisionOracle,
    count: int,
    seed: int,
    collision_margin_m: float,
    boundary_width_m: float,
    boundary_fraction: float,
    perturbation_std_rad: float,
    target_safe_fraction: float = 0.25,
) -> tuple[dict[str, np.ndarray], DatasetSummary]:
    if not 0.0 < target_safe_fraction < 1.0:
        raise ValueError("target_safe_fraction must be in (0, 1)")
    q_base = sample_periodic_configurations(count, seed)
    clearance_base = oracle.clearance(q_base)
    safe_target = int(round(count * target_safe_fraction))
    safe_q = [q_base[clearance_base >= collision_margin_m]]
    safe_clearance = [clearance_base[clearance_base >= collision_margin_m]]
    rng = np.random.default_rng(seed + 2)
    bootstrap_attempt = 0
    while sum(len(item) for item in safe_q) == 0 and bootstrap_attempt < 20:
        candidates = sample_periodic_configurations(count, seed + 10 + bootstrap_attempt)
        candidate_clearance = oracle.clearance(candidates)
        keep = candidate_clearance >= collision_margin_m
        if keep.any():
            safe_q.append(candidates[keep])
            safe_clearance.append(candidate_clearance[keep])
        bootstrap_attempt += 1
    if sum(len(item) for item in safe_q) == 0:
        raise RuntimeError("unable to bootstrap any safe configuration")
    attempts = 0
    while sum(len(item) for item in safe_q) < safe_target and attempts < 50:
        seeds = np.concatenate(safe_q, axis=0)
        needed = safe_target - len(seeds)
        selected = rng.choice(len(seeds), size=max(needed * 3, 256), replace=True)
        candidates = wrap_angles(
            seeds[selected]
            + rng.normal(
                0.0,
                min(perturbation_std_rad, 0.05),
                (len(selected), 6),
            )
        ).astype(np.float32)
        candidate_clearance = oracle.clearance(candidates)
        keep = candidate_clearance >= collision_margin_m
        if keep.any():
            safe_q.append(candidates[keep])
            safe_clearance.append(candidate_clearance[keep])
        attempts += 1
    all_safe_q = np.concatenate(safe_q, axis=0)
    all_safe_clearance = np.concatenate(safe_clearance, axis=0)
    if len(all_safe_q) < safe_target:
        raise RuntimeError(f"found only {len(all_safe_q)}/{safe_target} safe configurations")
    safe_selection = rng.choice(len(all_safe_q), size=safe_target, replace=False)

    collision_indices = np.flatnonzero(clearance_base < collision_margin_m)
    collision_target = count - safe_target
    boundary_target = min(int(round(count * boundary_fraction)), collision_target)
    boundary_order = collision_indices[
        np.argsort(np.abs(clearance_base[collision_indices] - collision_margin_m))
    ]
    boundary_selection = boundary_order[:boundary_target]
    remaining_target = collision_target - len(boundary_selection)
    remaining_pool = np.setdiff1d(collision_indices, boundary_selection, assume_unique=False)
    random_selection = rng.choice(
        remaining_pool,
        size=remaining_target,
        replace=len(remaining_pool) < remaining_target,
    )
    collision_selection = np.concatenate((boundary_selection, random_selection))
    q = np.concatenate((all_safe_q[safe_selection], q_base[collision_selection]), axis=0)
    clearance = np.concatenate(
        (all_safe_clearance[safe_selection], clearance_base[collision_selection]), axis=0
    )
    permutation = rng.permutation(count)
    q = q[permutation]
    clearance = clearance[permutation].astype(np.float32)
    collision = clearance < collision_margin_m
    summary = DatasetSummary(
        samples=len(q),
        collision_samples=int(collision.sum()),
        collision_ratio=float(collision.mean()),
        uniform_collision_ratio=float((clearance_base < collision_margin_m).mean()),
        boundary_samples=int(
            (np.abs(clearance - collision_margin_m) <= boundary_width_m).sum()
        ),
        clearance_min_m=float(clearance.min()),
        clearance_max_m=float(clearance.max()),
        seed=seed,
    )
    return {
        "q": q,
        "clearance_m": clearance,
        "collision": collision.astype(np.bool_),
    }, summary
