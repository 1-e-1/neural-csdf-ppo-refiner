"""Generate endpoint-safe random tasks without conditioning mid-path collisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import yaml

import _bootstrap

from h1_1111_grasp.trajectory_safety import environment_clearance, link_segments
from self_collision_refiner.geometry import ExactSelfCollisionOracle, wrap_angles
from src.planning.flow_postprocess import retime_sampled_path
from src.robot.torch_kinematics import DifferentiableZBotFK


def resolve(value: str) -> Path:
    return (_bootstrap.FULL64_ROOT / value).resolve()


def load_scenes(directory: Path, split: str) -> dict[str, np.ndarray]:
    shards = []
    for path in sorted(directory.glob(f"{split}_shard_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            shards.append(
                {
                    key: np.asarray(data[key])
                    for key in (
                        "sample_id",
                        "side",
                        "q_start",
                        "scene_points",
                        "scene_point_mask",
                    )
                }
            )
    if not shards:
        raise FileNotFoundError(f"no {split}_shard_*.npz under {directory}")
    return {
        key: np.concatenate([shard[key] for shard in shards], axis=0)
        for key in shards[0]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_bootstrap.FULL64_ROOT / "config_random_1000_v1.yaml",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = args.device or cfg["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    generation = cfg["random_task_generation"]
    geometry = cfg["geometry"]
    count = int(generation["tasks"])
    required_clearance = float(geometry["required_clearance_m"])
    environment_radius = float(geometry["environment_capsule_radius_m"])
    fk = DifferentiableZBotFK(resolve(cfg["models"]["kinematic_chain"])).to(device)
    oracle = ExactSelfCollisionOracle(
        fk,
        capsule_radius_m=float(geometry["capsule_radius_m"]),
        adjacency=int(geometry["adjacency"]),
        device=device,
        batch_size=int(geometry["exact_query_batch_size"]),
    )
    scenes = load_scenes(
        resolve(generation["source_dataset"]), str(generation["source_split"])
    )
    start_self = oracle.clearance(np.asarray(scenes["q_start"], dtype=np.float32))
    start_segments = link_segments(fk, scenes["q_start"], device=device)
    eligible_scenes = []
    start_environment = np.full(len(start_self), -np.inf, dtype=np.float64)
    for index in range(len(start_self)):
        cloud = scenes["scene_points"][index][scenes["scene_point_mask"][index]]
        start_environment[index] = environment_clearance(
            start_segments[index, 1:], cloud, capsule_radius_m=environment_radius
        )
        if (
            start_self[index] >= required_clearance
            and start_environment[index] >= required_clearance
        ):
            eligible_scenes.append(index)
    if not eligible_scenes:
        raise RuntimeError("no endpoint-safe source scenes")
    eligible_scenes = np.asarray(eligible_scenes, dtype=np.int64)

    accepted: dict[str, list] = {
        "source_id": [],
        "side": [],
        "q_start": [],
        "q_goal": [],
        "goal_position": [],
        "q": [],
        "duration": [],
        "scene_points": [],
        "scene_point_mask": [],
        "start_self_clearance_m": [],
        "goal_self_clearance_m": [],
        "start_environment_clearance_m": [],
        "goal_environment_clearance_m": [],
    }
    attempted = 0
    batch_size = int(generation.get("proposal_batch_size", 1024))
    max_batches = int(generation.get("maximum_attempt_batches", 100))
    joint_distance_min = float(generation.get("minimum_joint_distance_rad", 0.5))
    velocity_limits = np.asarray(cfg["limits"]["velocity"], dtype=np.float64)
    acceleration_limits = np.asarray(cfg["limits"]["acceleration"], dtype=np.float64)
    jerk_limits = np.asarray(cfg["limits"]["jerk"], dtype=np.float64)
    started = perf_counter()
    for _ in range(max_batches):
        if len(accepted["q"]) >= count:
            break
        scene_indices = rng.choice(eligible_scenes, size=batch_size, replace=True)
        q_start = np.asarray(scenes["q_start"][scene_indices], dtype=np.float64)
        q_goal = rng.uniform(-np.pi, np.pi, size=(batch_size, 6))
        delta = wrap_angles(q_goal - q_start)
        distance_ok = np.linalg.norm(delta, axis=1) >= joint_distance_min
        goal_self = oracle.clearance(q_goal.astype(np.float32))
        goal_segments = link_segments(fk, q_goal, device=device)
        attempted += batch_size
        for row in np.flatnonzero(distance_ok & (goal_self >= required_clearance)):
            scene_index = int(scene_indices[row])
            cloud = scenes["scene_points"][scene_index][
                scenes["scene_point_mask"][scene_index]
            ]
            goal_environment = environment_clearance(
                goal_segments[row, 1:], cloud, capsule_radius_m=environment_radius
            )
            if goal_environment < required_clearance:
                continue
            u = np.linspace(0.0, 1.0, 64)
            phase = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
            path = wrap_angles(q_start[row][None] + phase[:, None] * delta[row][None])
            _, timing = retime_sampled_path(
                path,
                predicted_duration=float(generation["min_duration_s"]),
                min_duration=float(generation["min_duration_s"]),
                nominal_max_duration=float(generation["nominal_max_duration_s"]),
                max_output_duration=float(generation["max_output_duration_s"]),
                velocity_limits=velocity_limits,
                acceleration_limits=acceleration_limits,
                jerk_limits=jerk_limits,
                timing_points=1024,
            )
            if not timing["duration_valid"]:
                continue
            with torch.no_grad():
                goal_position, _ = fk(
                    torch.as_tensor(q_goal[row], dtype=torch.float32, device=device)
                )
            accepted["source_id"].append(
                f"random_{len(accepted['q']):04d}_{scenes['sample_id'][scene_index]}"
            )
            accepted["side"].append(str(scenes["side"][scene_index]))
            accepted["q_start"].append(q_start[row].astype(np.float32))
            accepted["q_goal"].append(wrap_angles(q_goal[row]).astype(np.float32))
            accepted["goal_position"].append(goal_position.cpu().numpy().astype(np.float32))
            accepted["q"].append(path.astype(np.float32))
            accepted["duration"].append(np.float32(timing["duration_s"]))
            accepted["scene_points"].append(scenes["scene_points"][scene_index])
            accepted["scene_point_mask"].append(scenes["scene_point_mask"][scene_index])
            accepted["start_self_clearance_m"].append(np.float32(start_self[scene_index]))
            accepted["goal_self_clearance_m"].append(np.float32(goal_self[row]))
            accepted["start_environment_clearance_m"].append(
                np.float32(start_environment[scene_index])
            )
            accepted["goal_environment_clearance_m"].append(
                np.float32(goal_environment)
            )
            if len(accepted["q"]) >= count:
                break
        print(
            f"[RANDOM-ENDPOINTS] accepted={len(accepted['q'])}/{count} "
            f"attempted={attempted}",
            flush=True,
        )
    if len(accepted["q"]) < count:
        raise RuntimeError(
            f"generated only {len(accepted['q'])}/{count} tasks after {attempted} proposals"
        )

    output = {
        key: np.asarray(values[:count]) for key, values in accepted.items()
    }
    output_path = resolve(generation["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **output)
    report = {
        "method": "independent uniform endpoint sampling with no mid-path collision conditioning",
        "seed": seed,
        "tasks": count,
        "attempted_endpoint_pairs": attempted,
        "acceptance_rate": count / attempted,
        "source_split": str(generation["source_split"]),
        "reachable_by_construction": True,
        "endpoint_criteria": {
            "self_clearance_m_min": required_clearance,
            "reconstructed_environment_clearance_m_min": required_clearance,
            "duration_validity_max_s": float(generation["max_output_duration_s"]),
        },
        "mid_path_collision_conditioning": "none",
        "elapsed_s": perf_counter() - started,
        "output": str(output_path),
    }
    report_path = resolve(generation["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
