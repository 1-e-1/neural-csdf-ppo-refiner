"""Generate fixed endpoint-safe, mid-path self-collision tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import yaml

import _bootstrap

from self_collision_refiner.geometry import (
    ExactSelfCollisionOracle,
    periodic_resample,
    unwrap_trajectory,
    wrap_angles,
)
from src.robot.torch_kinematics import DifferentiableZBotFK


def resolve(value: str) -> Path:
    return (_bootstrap.FULL64_ROOT / value).resolve()


def load_split(directory: Path, split: str) -> dict[str, np.ndarray]:
    shards = []
    for path in sorted(directory.glob(f"{split}_shard_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            shards.append(
                {
                    "q": np.asarray(data["q"], dtype=np.float32),
                    "duration": np.asarray(data["duration"], dtype=np.float32),
                    "sample_id": np.asarray(data["sample_id"]),
                }
            )
    if not shards:
        raise FileNotFoundError(f"no {split}_shard_*.npz under {directory}")
    return {
        key: np.concatenate([shard[key] for shard in shards], axis=0)
        for key in shards[0]
    }


def dense_clearance(
    paths: np.ndarray, oracle: ExactSelfCollisionOracle
) -> tuple[np.ndarray, np.ndarray]:
    dense = np.stack([periodic_resample(path, 256) for path in paths])
    clearance = oracle.clearance(dense.reshape(-1, 6)).reshape(len(paths), 256)
    return dense, clearance


def safe_source_indices(
    paths: np.ndarray,
    oracle: ExactSelfCollisionOracle,
    margin_m: float,
    batch_size: int,
) -> np.ndarray:
    selected = []
    for start in range(0, len(paths), batch_size):
        end = min(start + batch_size, len(paths))
        _, clearance = dense_clearance(paths[start:end], oracle)
        selected.extend(
            (start + np.flatnonzero(clearance.min(axis=1) >= margin_m)).tolist()
        )
    return np.asarray(selected, dtype=np.int64)


def generate_tasks(
    source: dict[str, np.ndarray],
    count: int,
    oracle: ExactSelfCollisionOracle,
    cfg: dict,
    margin_m: float,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict]:
    rng = np.random.default_rng(seed)
    batch_size = int(cfg["proposal_batch_size"])
    safe_indices = safe_source_indices(
        source["q"], oracle, margin_m=margin_m, batch_size=batch_size
    )
    if len(safe_indices) == 0:
        raise RuntimeError("source split has no continuously safe trajectories")
    amplitude_min, amplitude_max = map(float, cfg["perturbation_amplitude_rad"])
    radius_min, radius_max = map(int, cfg["perturbation_radius_points"])
    clearance_min, clearance_max = map(float, cfg["accepted_clearance_range_m"])
    collected: dict[str, list[np.ndarray]] = {
        "q": [],
        "q_safe_reference": [],
        "duration": [],
        "source_id": [],
        "initial_min_clearance_m": [],
        "collision_index": [],
    }
    attempted = 0
    for _ in range(int(cfg["maximum_attempt_batches"])):
        if len(collected["q"]) >= count:
            break
        chosen = rng.choice(safe_indices, size=batch_size, replace=True)
        reference = source["q"][chosen]
        continuous = np.stack([unwrap_trajectory(path) for path in reference])
        centers = rng.integers(8, 56, size=batch_size)
        radii = rng.integers(radius_min, radius_max + 1, size=batch_size)
        amplitudes = rng.uniform(amplitude_min, amplitude_max, size=batch_size)
        directions = rng.normal(size=(batch_size, 6))
        active = rng.random((batch_size, 6)) < 0.5
        empty = ~active.any(axis=1)
        active[empty, rng.integers(0, 6, size=int(empty.sum()))] = True
        directions *= active
        directions /= np.linalg.norm(directions, axis=1, keepdims=True).clip(1.0e-8)
        index = np.arange(64)[None]
        relative = (index - centers[:, None]) / radii[:, None]
        support = np.abs(relative) <= 1.0
        bump = np.where(support, 0.5 * (1.0 + np.cos(np.pi * relative)), 0.0)
        bump[:, 0] = 0.0
        bump[:, -1] = 0.0
        candidate = wrap_angles(
            continuous
            + bump[:, :, None]
            * amplitudes[:, None, None]
            * directions[:, None, :]
        ).astype(np.float32)
        _, clearance = dense_clearance(candidate, oracle)
        minima = clearance.min(axis=1)
        collision_indices = clearance.argmin(axis=1)
        endpoint_safe = (
            (clearance[:, 0] >= margin_m) & (clearance[:, -1] >= margin_m)
        )
        accepted = np.flatnonzero(
            endpoint_safe
            & (minima >= clearance_min)
            & (minima <= clearance_max)
            & (collision_indices > 0)
            & (collision_indices < 255)
        )
        attempted += batch_size
        for row in accepted:
            collected["q"].append(candidate[row])
            collected["q_safe_reference"].append(reference[row])
            collected["duration"].append(source["duration"][chosen[row]])
            collected["source_id"].append(source["sample_id"][chosen[row]])
            collected["initial_min_clearance_m"].append(minima[row])
            collected["collision_index"].append(collision_indices[row])
            if len(collected["q"]) >= count:
                break
    if len(collected["q"]) < count:
        raise RuntimeError(
            f"generated only {len(collected['q'])}/{count} tasks after {attempted} proposals"
        )
    output = {
        "q": np.asarray(collected["q"][:count], dtype=np.float32),
        "q_safe_reference": np.asarray(
            collected["q_safe_reference"][:count], dtype=np.float32
        ),
        "duration": np.asarray(collected["duration"][:count], dtype=np.float32),
        "source_id": np.asarray(collected["source_id"][:count]),
        "initial_min_clearance_m": np.asarray(
            collected["initial_min_clearance_m"][:count], dtype=np.float32
        ),
        "collision_index": np.asarray(
            collected["collision_index"][:count], dtype=np.int32
        ),
    }
    summary = {
        "tasks": count,
        "source_trajectories": len(source["q"]),
        "continuously_safe_source_trajectories": len(safe_indices),
        "attempted_proposals": attempted,
        "acceptance_rate": count / attempted,
        "min_clearance_m": float(output["initial_min_clearance_m"].min()),
        "mean_clearance_m": float(output["initial_min_clearance_m"].mean()),
        "max_clearance_m": float(output["initial_min_clearance_m"].max()),
        "endpoint_safe_tasks": count,
        "midpath_collision_tasks": count,
        "seed": seed,
    }
    return output, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=_bootstrap.FULL64_ROOT / "config.yaml"
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = args.device or cfg["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    seed = int(cfg["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    geometry = cfg["geometry"]
    fk = DifferentiableZBotFK(resolve(cfg["models"]["kinematic_chain"])).to(device)
    oracle = ExactSelfCollisionOracle(
        fk,
        capsule_radius_m=float(geometry["capsule_radius_m"]),
        adjacency=int(geometry["adjacency"]),
        device=device,
        batch_size=int(geometry["exact_query_batch_size"]),
    )
    task_cfg = cfg["task_generation"]
    source_dir = resolve(task_cfg["source_dataset"])
    started = perf_counter()
    reports = {}
    for split_name, count_key, output_key, offset in (
        (task_cfg["train_split"], "train_tasks", "train_output", 0),
        (task_cfg["test_split"], "test_tasks", "test_output", 10000),
    ):
        source = load_split(source_dir, str(split_name))
        tasks, report = generate_tasks(
            source=source,
            count=int(task_cfg[count_key]),
            oracle=oracle,
            cfg=task_cfg,
            margin_m=float(geometry["required_clearance_m"]),
            seed=seed + offset,
        )
        output = resolve(task_cfg[output_key])
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **tasks)
        reports[str(split_name)] = {"output": str(output), **report}
        print(
            f"[TASKS] split={split_name} tasks={report['tasks']} "
            f"mean_clearance={report['mean_clearance_m']:.5f}",
            flush=True,
        )
    final_report = {
        "method": "compact-support interior perturbation of continuously safe 64-point trajectories",
        "endpoint_policy": "first and last joint states are unchanged and exact-safe",
        "train_test_separation": "train tasks use train split; test tasks use test split",
        "elapsed_s": perf_counter() - started,
        "splits": reports,
    }
    report_path = resolve(task_cfg["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(final_report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

