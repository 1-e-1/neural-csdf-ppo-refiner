"""Evaluate full-64-point PPO on a fixed held-out mid-path collision set."""

from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import yaml

import _bootstrap

from h1_1111_grasp.trajectory_safety import environment_clearance_batch, link_segments
from self_collision_refiner.geometry import (
    ExactSelfCollisionOracle,
    adaptive_validate_trajectory,
    periodic_resample,
    unwrap_trajectory,
)
from self_collision_refiner.models import NeuralCSDF, TrajectoryActorCritic
from self_collision_refiner.trajectory_env import TrajectoryRefinementEnv
from src.planning.flow_postprocess import retime_sampled_path
from src.robot.torch_kinematics import DifferentiableZBotFK


def resolve(value: str) -> Path:
    return (_bootstrap.FULL64_ROOT / value).resolve()


def wilson(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    radius = z / denominator * sqrt(
        probability * (1.0 - probability) / total
        + z * z / (4.0 * total * total)
    )
    return [center - radius, center + radius]


def stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def derivative_metrics(
    paths: np.ndarray,
    durations: np.ndarray,
    velocity_limits: np.ndarray,
    acceleration_limits: np.ndarray,
    jerk_limits: np.ndarray,
    reference_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    acceleration_values = []
    jerk_values = []
    dynamics_ok = []
    for path, duration in zip(paths, durations):
        if reference_points is not None and len(path) != reference_points:
            path = periodic_resample(path, reference_points)
        continuous = unwrap_trajectory(path)
        time = np.linspace(0.0, float(duration), len(path))
        velocity = np.gradient(continuous, time, axis=0, edge_order=2)
        acceleration = np.gradient(velocity, time, axis=0, edge_order=2)
        jerk = np.gradient(acceleration, time, axis=0, edge_order=2)
        acceleration_values.append(float(np.sqrt(np.mean(acceleration**2))))
        jerk_values.append(float(np.sqrt(np.mean(jerk**2))))
        dynamics_ok.append(
            bool(
                (
                    np.max(np.abs(velocity), axis=0)
                    <= velocity_limits * (1.0 + 1.0e-6)
                ).all()
                and (
                    np.max(np.abs(acceleration), axis=0)
                    <= acceleration_limits * (1.0 + 1.0e-6)
                ).all()
                and (
                    np.max(np.abs(jerk), axis=0)
                    <= jerk_limits * (1.0 + 1.0e-6)
                ).all()
            )
        )
    return (
        np.asarray(acceleration_values),
        np.asarray(jerk_values),
        np.asarray(dynamics_ok, dtype=np.bool_),
    )


def environment_clearances(
    paths: np.ndarray,
    scene_points: np.ndarray,
    scene_point_masks: np.ndarray,
    fk_model: torch.nn.Module,
    device: str,
    capsule_radius_m: float,
    reference_points: int,
    batch_size: int,
    label: str,
) -> np.ndarray:
    values = np.empty(len(paths), dtype=np.float64)
    for start in range(0, len(paths), batch_size):
        end = min(start + batch_size, len(paths))
        sampled = np.stack(
            [
                periodic_resample(path, reference_points)
                if len(path) != reference_points
                else path
                for path in paths[start:end]
            ]
        )
        segments = link_segments(
            fk_model, sampled.reshape(-1, 6), device=device
        ).reshape(end - start, reference_points, -1, 2, 3)
        for local_index, global_index in enumerate(range(start, end)):
            cloud = scene_points[global_index][scene_point_masks[global_index]]
            values[global_index] = environment_clearance_batch(
                segments[local_index : local_index + 1, :, 1:],
                cloud,
                capsule_radius_m=capsule_radius_m,
            )[0]
        print(f"[{label}] {end}/{len(paths)}", flush=True)
    return values


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

    evaluation = cfg["evaluation"]
    input_path = resolve(evaluation["input"])
    with np.load(input_path, allow_pickle=False) as data:
        test = {key: np.asarray(data[key]) for key in data.files}
    paths = np.asarray(test["q"], dtype=np.float32)
    durations = np.asarray(test["duration"], dtype=np.float32)
    csdf_path = resolve(cfg["models"]["csdf_checkpoint"])
    csdf_checkpoint = torch.load(csdf_path, map_location=device, weights_only=False)
    csdf = NeuralCSDF(tuple(csdf_checkpoint["hidden_dims"])).to(device)
    csdf.load_state_dict(csdf_checkpoint["model_state_dict"])
    csdf.eval()
    checkpoint_path = resolve(cfg["ppo"]["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    refiner_cfg = checkpoint["refiner_config"]
    ppo_cfg = checkpoint["ppo_config"]
    env = TrajectoryRefinementEnv(paths, csdf, refiner_cfg, device, seed)
    policy = TrajectoryActorCritic(
        state_dim=int(checkpoint["state_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        hidden_dim=int(ppo_cfg["hidden_dim"]),
        initial_std=float(ppo_cfg["initial_std"]),
    ).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    outputs = []
    neural_initial = []
    neural_final = []
    batch_size = int(evaluation["batch_size"])
    inference_started = perf_counter()
    for start in range(0, len(paths), batch_size):
        end = min(start + batch_size, len(paths))
        state, initial = env.reset(paths=paths[start:end], add_corruption=False)
        with torch.no_grad():
            for _ in range(env.episode_steps):
                action, _, _ = policy.act(state, deterministic=True)
                state, _, _, final = env.step(action)
        outputs.append(env.trajectories())
        neural_initial.append(initial["min_clearance_m"].cpu().numpy())
        neural_final.append(final["min_clearance_m"].cpu().numpy())
    inference_elapsed = perf_counter() - inference_started
    refined = np.concatenate(outputs, axis=0)
    output_points = env.output_points
    initial_dense = np.stack(
        [periodic_resample(path, output_points) for path in paths]
    )
    limit_cfg = yaml.safe_load(
        resolve(evaluation["robot_limits"]).read_text(encoding="utf-8")
    )["planning_soft_limits"]
    velocity_limits = np.asarray(limit_cfg["velocity"], dtype=np.float64)
    acceleration_limits = np.asarray(limit_cfg["acceleration"], dtype=np.float64)
    jerk_limits = np.asarray(limit_cfg["jerk"], dtype=np.float64)
    time_scaling_cfg = evaluation.get("time_scaling", {})
    final_durations = durations.astype(np.float64).copy()
    duration_valid = np.ones(len(paths), dtype=np.bool_)
    duration_exceeds_nominal = np.zeros(len(paths), dtype=np.bool_)
    if bool(time_scaling_cfg.get("retime_after_refinement", False)):
        timing_points = int(time_scaling_cfg.get("timing_points", output_points))
        if timing_points != output_points:
            raise ValueError(
                "evaluation.time_scaling.timing_points must equal refiner output_points"
            )
        retimed = []
        for index, path in enumerate(refined):
            trajectory, timing = retime_sampled_path(
                q=path,
                predicted_duration=float(durations[index]),
                min_duration=float(time_scaling_cfg.get("min_duration_s", 0.8)),
                nominal_max_duration=float(
                    time_scaling_cfg.get("nominal_max_duration_s", 5.0)
                ),
                max_output_duration=float(
                    time_scaling_cfg.get("max_output_duration_s", 30.0)
                ),
                velocity_limits=velocity_limits,
                acceleration_limits=acceleration_limits,
                jerk_limits=jerk_limits,
                timing_points=timing_points,
            )
            retimed.append(trajectory.q)
            final_durations[index] = trajectory.duration
            duration_valid[index] = bool(timing["duration_valid"])
            duration_exceeds_nominal[index] = bool(
                timing["duration_exceeds_nominal_max"]
            )
        refined = np.asarray(retimed, dtype=np.float32)

    geometry = cfg["geometry"]
    fk = DifferentiableZBotFK(resolve(cfg["models"]["kinematic_chain"])).to(device)
    oracle = ExactSelfCollisionOracle(
        fk,
        capsule_radius_m=float(geometry["capsule_radius_m"]),
        adjacency=int(geometry["adjacency"]),
        device=device,
        batch_size=int(geometry["exact_query_batch_size"]),
    )
    validation_args = {
        "oracle": oracle,
        "required_clearance_m": float(geometry["required_clearance_m"]),
        "max_joint_step_rad": float(geometry["continuous_check_max_joint_step_rad"]),
        "near_boundary_extra_subdivisions": int(
            geometry["near_boundary_extra_subdivisions"]
        ),
    }
    initial_reports = []
    final_reports = []
    for index in range(len(paths)):
        initial_reports.append(
            adaptive_validate_trajectory(initial_dense[index], **validation_args)
        )
        final_reports.append(
            adaptive_validate_trajectory(refined[index], **validation_args)
        )
        if (index + 1) % 64 == 0 or index + 1 == len(paths):
            print(f"[EVAL64] {index + 1}/{len(paths)}", flush=True)
    initial_clearance = np.asarray([item.min_clearance_m for item in initial_reports])
    final_clearance = np.asarray([item.min_clearance_m for item in final_reports])
    exact_success = np.asarray([item.safe for item in final_reports], dtype=np.bool_)
    endpoints_preserved = np.all(
        np.isclose(refined[:, [0, -1]], paths[:, [0, -1]], atol=1.0e-5),
        axis=(1, 2),
    )
    improved = final_clearance > initial_clearance
    dynamics_reference_points = int(
        evaluation.get("dynamics_reference_points", output_points)
    )
    initial_acceleration, initial_jerk, initial_dynamics_ok = derivative_metrics(
        initial_dense,
        durations,
        velocity_limits,
        acceleration_limits,
        jerk_limits,
        dynamics_reference_points,
    )
    final_acceleration, final_jerk, final_dynamics_ok = derivative_metrics(
        refined,
        final_durations,
        velocity_limits,
        acceleration_limits,
        jerk_limits,
        dynamics_reference_points,
    )
    has_environment_geometry = {
        "scene_points",
        "scene_point_mask",
    }.issubset(test)
    if has_environment_geometry:
        environment_points = int(
            evaluation.get("environment_check_points", output_points)
        )
        environment_radius = float(
            geometry.get("environment_capsule_radius_m", 0.05)
        )
        initial_environment = environment_clearances(
            initial_dense,
            test["scene_points"],
            np.asarray(test["scene_point_mask"], dtype=np.bool_),
            fk,
            device,
            environment_radius,
            environment_points,
            batch_size,
            "ENV-INPUT",
        )
        final_environment = environment_clearances(
            refined,
            test["scene_points"],
            np.asarray(test["scene_point_mask"], dtype=np.bool_),
            fk,
            device,
            environment_radius,
            environment_points,
            batch_size,
            "ENV-OUTPUT",
        )
    else:
        initial_environment = np.full(len(paths), np.inf)
        final_environment = np.full(len(paths), np.inf)
    required_clearance = float(geometry["required_clearance_m"])
    initial_self_safe = initial_clearance >= required_clearance
    final_self_safe = exact_success
    initial_environment_safe = initial_environment >= required_clearance
    final_environment_safe = final_environment >= required_clearance
    initial_any_safe = initial_self_safe & initial_environment_safe
    final_any_safe = final_self_safe & final_environment_safe
    initial_self_collision_tasks = int((~initial_self_safe).sum())
    exact_repair_mask = (~initial_self_safe) & final_self_safe
    successes = int(exact_repair_mask.sum())
    usable_success = (
        final_any_safe
        & final_dynamics_ok
        & endpoints_preserved
        & duration_valid
    )
    report = {
        "method": "full 64-point input PPO; 64 controls; 62 interior C-SDF gradient-gate actions",
        "evaluation_policy": "single fixed checkpoint and held-out test split; no automatic redesign or retraining",
        "seed": seed,
        "test_tasks": len(paths),
        "input_contract": {
            "trajectory_shape": [64, 6],
            "observation_dim": env.state_dim,
            "control_points": env.control_count,
            "action_dim": env.action_dim,
            "episode_steps": env.episode_steps,
            "output_shape": [output_points, 6],
            "dynamics_reference_points": dynamics_reference_points,
            "duration_shape": [len(paths)],
        },
        "task_contract": {
            "endpoint_safe_tasks": 0,
            "initial_exact_unsafe_tasks": int((initial_clearance < float(geometry["required_clearance_m"])).sum()),
            "train_test_source_split_separated": True,
        },
        "performance": {
            "exact_repair_successes": successes,
            "exact_repair_eligible_tasks": initial_self_collision_tasks,
            "exact_repair_success_rate": successes
            / max(initial_self_collision_tasks, 1),
            "exact_repair_success_wilson_95": wilson(
                successes, initial_self_collision_tasks
            ),
            "final_exact_self_safe_tasks": int(final_self_safe.sum()),
            "clearance_improved_tasks": int(improved.sum()),
            "clearance_improved_rate": float(improved.mean()),
            "endpoint_preserved_tasks": int(endpoints_preserved.sum()),
            "input_dynamics_passes": int(initial_dynamics_ok.sum()),
            "input_dynamics_pass_rate": float(initial_dynamics_ok.mean()),
            "output_dynamics_passes": int(final_dynamics_ok.sum()),
            "output_dynamics_pass_rate": float(final_dynamics_ok.mean()),
            "duration_valid_tasks": int(duration_valid.sum()),
            "duration_exceeds_nominal_tasks": int(duration_exceeds_nominal.sum()),
            "safe_low_impact_successes": int(usable_success.sum()),
            "safe_low_impact_success_rate": float(usable_success.mean()),
            "safe_low_impact_success_wilson_95": wilson(
                int(usable_success.sum()), len(paths)
            ),
            "neural_csdf_proxy_successes": int(
                (np.concatenate(neural_final) >= float(geometry["required_clearance_m"])).sum()
            ),
        },
        "clearance_m": {
            "initial_exact": stats(initial_clearance),
            "final_exact": stats(final_clearance),
            "change_exact": stats(final_clearance - initial_clearance),
            "initial_neural": stats(np.concatenate(neural_initial)),
            "final_neural": stats(np.concatenate(neural_final)),
        },
        "smoothness": {
            "initial_acceleration_rms_rad_s2": stats(initial_acceleration),
            "final_acceleration_rms_rad_s2": stats(final_acceleration),
            "initial_jerk_rms_rad_s3": stats(initial_jerk),
            "final_jerk_rms_rad_s3": stats(final_jerk),
        },
        "time_scaling": {
            "retime_after_refinement": bool(
                time_scaling_cfg.get("retime_after_refinement", False)
            ),
            "timing_points": int(
                time_scaling_cfg.get("timing_points", output_points)
            ),
            "nominal_max_duration_s": float(
                time_scaling_cfg.get("nominal_max_duration_s", 5.0)
            ),
            "max_output_duration_s": float(
                time_scaling_cfg.get("max_output_duration_s", 30.0)
            ),
            "hard_clip_applied": False,
            "final_duration_s": stats(final_durations),
        },
        "inference": {
            "batch_size": batch_size,
            "total_s": inference_elapsed,
            "mean_per_task_ms": 1000.0 * inference_elapsed / len(paths),
        },
        "checkpoint": str(checkpoint_path),
        "input_dataset": str(input_path),
    }
    if has_environment_geometry:
        initially_unsafe = ~initial_any_safe
        initially_safe = initial_any_safe
        report["natural_random_task_performance"] = {
            "mid_path_collision_was_not_conditioned": True,
            "initial_self_collision_tasks": int((~initial_self_safe).sum()),
            "initial_environment_or_mutual_collision_tasks": int(
                (~initial_environment_safe).sum()
            ),
            "initial_any_collision_tasks": int(initially_unsafe.sum()),
            "initial_collision_free_tasks": int(initially_safe.sum()),
            "ppo_self_collision_repairs": int(
                ((~initial_self_safe) & final_self_safe).sum()
            ),
            "ppo_self_collision_damage": int(
                (initial_self_safe & (~final_self_safe)).sum()
            ),
            "ppo_environment_collision_repairs": int(
                ((~initial_environment_safe) & final_environment_safe).sum()
            ),
            "ppo_environment_collision_damage": int(
                (initial_environment_safe & (~final_environment_safe)).sum()
            ),
            "ppo_any_collision_repairs": int(
                (initially_unsafe & final_any_safe).sum()
            ),
            "ppo_any_collision_repair_rate": float(
                (initially_unsafe & final_any_safe).sum()
                / max(int(initially_unsafe.sum()), 1)
            ),
            "initially_safe_preserved": int(
                (initially_safe & final_any_safe).sum()
            ),
            "initially_safe_preservation_rate": float(
                (initially_safe & final_any_safe).sum()
                / max(int(initially_safe.sum()), 1)
            ),
            "final_self_safe_tasks": int(final_self_safe.sum()),
            "final_environment_safe_tasks": int(final_environment_safe.sum()),
            "final_collision_free_tasks": int(final_any_safe.sum()),
            "final_safe_low_impact_tasks": int(usable_success.sum()),
            "environment_check_points": environment_points,
        }
        report["environment_clearance_m"] = {
            "initial": stats(initial_environment),
            "final": stats(final_environment),
            "change": stats(final_environment - initial_environment),
        }
    endpoint_clearance = oracle.clearance(
        paths[:, [0, -1]].reshape(-1, 6)
    ).reshape(len(paths), 2)
    report["task_contract"]["endpoint_safe_tasks"] = int(
        (endpoint_clearance >= float(geometry["required_clearance_m"])).all(axis=1).sum()
    )
    output_path = resolve(evaluation["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        source_id=test["source_id"],
        q_input=paths,
        q=refined,
        duration=final_durations.astype(np.float32),
        source_duration=durations,
        duration_valid=duration_valid,
        exact_initial_clearance_m=initial_clearance.astype(np.float32),
        exact_final_clearance_m=final_clearance.astype(np.float32),
        exact_repair_success=exact_success,
        initial_self_safe=initial_self_safe,
        final_self_safe=final_self_safe,
        initial_environment_clearance_m=initial_environment.astype(np.float32),
        final_environment_clearance_m=final_environment.astype(np.float32),
        initial_collision_free=initial_any_safe,
        final_collision_free=final_any_safe,
        dynamics_ok=final_dynamics_ok,
        safe_low_impact_success=usable_success,
        endpoint_preserved=endpoints_preserved,
    )
    report_path = resolve(evaluation["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
