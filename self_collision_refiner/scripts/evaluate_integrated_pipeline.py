"""Evaluate frozen five-view Visual CFM outputs with the self-collision refiner."""

from __future__ import annotations

import argparse
import hashlib
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
    project_terminal_with_minimum_jerk,
    refine_safe_terminal_configurations_batch,
    unwrap_trajectory,
)
from self_collision_refiner.models import NeuralCSDF, TrajectoryActorCritic
from self_collision_refiner.trajectory_env import TrajectoryRefinementEnv
from src.planning.flow_postprocess import retime_sampled_path
from src.robot.torch_kinematics import DifferentiableZBotFK


def resolve(value: str) -> Path:
    return (_bootstrap.SUBPROJECT_ROOT / value).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    return [max(0.0, center - radius), min(1.0, center + radius)]


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def side_success_rate(records: list[dict], side: str) -> float | None:
    selected = [record["success"] for record in records if record["side"] == side]
    return float(np.mean(selected)) if selected else None


def trajectory_metrics(
    q: np.ndarray,
    duration_s: float,
    velocity_limits: np.ndarray,
    acceleration_limits: np.ndarray,
    jerk_limits: np.ndarray,
    reference_points: int | None = None,
) -> dict[str, float | bool]:
    if reference_points is not None and len(q) != reference_points:
        q = periodic_resample(q, reference_points)
    continuous = unwrap_trajectory(q)
    time = np.linspace(0.0, float(duration_s), len(continuous))
    velocity = np.gradient(continuous, time, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, time, axis=0, edge_order=2)
    jerk = np.gradient(acceleration, time, axis=0, edge_order=2)
    velocity_peak = np.max(np.abs(velocity), axis=0)
    acceleration_peak = np.max(np.abs(acceleration), axis=0)
    jerk_peak = np.max(np.abs(jerk), axis=0)
    return {
        "velocity_rms_rad_s": float(np.sqrt(np.mean(velocity**2))),
        "acceleration_rms_rad_s2": float(np.sqrt(np.mean(acceleration**2))),
        "jerk_rms_rad_s3": float(np.sqrt(np.mean(jerk**2))),
        "max_velocity_ratio": float(np.max(velocity_peak / velocity_limits)),
        "max_acceleration_ratio": float(
            np.max(acceleration_peak / acceleration_limits)
        ),
        "max_jerk_ratio": float(np.max(jerk_peak / jerk_limits)),
        "dynamics_ok": bool(
            (velocity_peak <= velocity_limits * (1.0 + 1.0e-6)).all()
            and (
                acceleration_peak <= acceleration_limits * (1.0 + 1.0e-6)
            ).all()
            and (jerk_peak <= jerk_limits * (1.0 + 1.0e-6)).all()
        ),
    }


def load_test_rows(data_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    rows: dict[str, dict[str, np.ndarray]] = {}
    for path in sorted(data_dir.glob("test_shard_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            sample_ids = data["sample_id"]
            scene_points = data["scene_points"]
            scene_masks = data["scene_point_mask"]
            for index, sample_id in enumerate(sample_ids):
                key = str(sample_id)
                if key in rows:
                    raise ValueError(f"duplicate test sample_id: {key}")
                rows[key] = {
                    "scene_points": np.asarray(scene_points[index]),
                    "scene_point_mask": np.asarray(scene_masks[index], dtype=np.bool_),
                }
    if not rows:
        raise FileNotFoundError(f"no test shards under {data_dir}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_bootstrap.SUBPROJECT_ROOT / "integrated_evaluation.yaml",
    )
    parser.add_argument("--max_tasks", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = args.device or cfg["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    seed = int(cfg["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    source_cfg = cfg["sources"]
    source_path = resolve(source_cfg["visual_selected"])
    visual_report_path = resolve(source_cfg["visual_report"])
    visual_checkpoint = resolve(source_cfg["visual_checkpoint"])
    for required in (source_path, visual_report_path, visual_checkpoint):
        if not required.is_file():
            raise FileNotFoundError(required)
    with np.load(source_path, allow_pickle=False) as data:
        source = {key: np.asarray(data[key]) for key in data.files}
    task_count = len(source["q"])
    if args.max_tasks:
        task_count = min(task_count, args.max_tasks)
        source = {key: value[:task_count] for key, value in source.items()}
    if source["q"].shape != (task_count, 64, 6):
        raise ValueError(f"expected Visual CFM q [N,64,6], got {source['q'].shape}")
    visual_report = json.loads(visual_report_path.read_text(encoding="utf-8"))
    visual_records = {
        str(record["sample_id"]): record for record in visual_report["records"]
    }
    test_rows = load_test_rows(resolve(source_cfg["test_dataset"]))
    for sample_id in source["sample_id"]:
        key = str(sample_id)
        if key not in visual_records or key not in test_rows:
            raise KeyError(f"missing Visual CFM report or test geometry for {key}")

    started = perf_counter()
    model_cfg = cfg["models"]
    csdf_checkpoint_path = resolve(model_cfg["csdf_checkpoint"])
    refiner_checkpoint_path = resolve(model_cfg["refiner_checkpoint"])
    csdf_checkpoint = torch.load(
        csdf_checkpoint_path, map_location=device, weights_only=False
    )
    csdf = NeuralCSDF(tuple(csdf_checkpoint["hidden_dims"])).to(device)
    csdf.load_state_dict(csdf_checkpoint["model_state_dict"])
    csdf.eval()
    fk = DifferentiableZBotFK(resolve(model_cfg["kinematic_chain"])).to(device)
    evaluation = cfg["evaluation"]
    required_clearance = float(evaluation["required_clearance_m"])
    oracle = ExactSelfCollisionOracle(
        fk,
        capsule_radius_m=float(evaluation["capsule_radius_m"]),
        adjacency=int(evaluation["adjacency"]),
        device=device,
        batch_size=int(evaluation["exact_query_batch_size"]),
    )

    q64 = np.asarray(source["q"], dtype=np.float32)
    goals = np.asarray(source["goal_position"], dtype=np.float32)
    endpoint_clearance = oracle.clearance(q64[:, [0, -1]].reshape(-1, 6)).reshape(
        task_count, 2
    )
    terminal_indices = np.flatnonzero(
        (endpoint_clearance[:, 0] >= required_clearance)
        & (endpoint_clearance[:, 1] < required_clearance)
    )
    q_projected = q64.copy()
    terminal_results: dict[int, object] = {}
    projection_cfg = cfg["terminal_projection"]
    if len(terminal_indices):
        results = refine_safe_terminal_configurations_batch(
            fk_model=fk,
            csdf_model=csdf,
            oracle=oracle,
            q_initial=q64[terminal_indices, -1],
            target_positions=goals[terminal_indices],
            required_clearance_m=required_clearance,
            device=device,
            restarts=int(projection_cfg["restarts"]),
            iterations=int(projection_cfg["iterations"]),
            position_tolerance_m=float(projection_cfg["position_tolerance_m"]),
            seed=seed,
        )
        failed_positions = []
        for local_index, result in enumerate(results):
            source_index = int(terminal_indices[local_index])
            terminal_results[source_index] = result
            if result.found:
                q_projected[source_index] = project_terminal_with_minimum_jerk(
                    q64[source_index], result.q
                )
            else:
                failed_positions.append(local_index)
        if failed_positions:
            failed_source_indices = terminal_indices[failed_positions]
            fallback = refine_safe_terminal_configurations_batch(
                fk_model=fk,
                csdf_model=csdf,
                oracle=oracle,
                q_initial=q64[failed_source_indices, -1],
                target_positions=goals[failed_source_indices],
                required_clearance_m=required_clearance,
                device=device,
                restarts=int(projection_cfg["fallback_restarts"]),
                iterations=int(projection_cfg["fallback_iterations"]),
                position_tolerance_m=float(projection_cfg["position_tolerance_m"]),
                seed=seed + 10000,
            )
            for source_index, result in zip(failed_source_indices, fallback):
                source_index = int(source_index)
                terminal_results[source_index] = result
                if result.found:
                    q_projected[source_index] = project_terminal_with_minimum_jerk(
                        q64[source_index], result.q
                    )
    policy_checkpoint = torch.load(
        refiner_checkpoint_path, map_location=device, weights_only=False
    )
    refiner_config_path = resolve(model_cfg.get("refiner_config", "config.yaml"))
    refiner_file_cfg = yaml.safe_load(
        refiner_config_path.read_text(encoding="utf-8")
    )
    refiner_cfg = refiner_file_cfg["refiner"]
    output_points = int(refiner_cfg["output_points"])
    limits_cfg = yaml.safe_load(
        resolve(source_cfg["robot_limits"]).read_text(encoding="utf-8")
    )["planning_soft_limits"]
    velocity_limits = np.asarray(limits_cfg["velocity"], dtype=np.float64)
    acceleration_limits = np.asarray(limits_cfg["acceleration"], dtype=np.float64)
    jerk_limits = np.asarray(limits_cfg["jerk"], dtype=np.float64)
    time_scaling_cfg = evaluation.get("time_scaling", {})
    timing_points = int(time_scaling_cfg.get("timing_points", output_points))
    if timing_points != output_points:
        raise ValueError(
            "integrated time_scaling.timing_points must equal refiner output_points"
        )
    min_duration = float(time_scaling_cfg.get("min_duration_s", 0.8))
    nominal_max_duration = float(
        time_scaling_cfg.get("nominal_max_duration_s", 5.0)
    )
    max_output_duration = float(
        time_scaling_cfg.get("max_output_duration_s", 30.0)
    )
    projection_durations = np.asarray(source["duration"], dtype=np.float64).copy()
    projection_duration_valid = np.ones(task_count, dtype=np.bool_)
    projection_duration_exceeds_nominal = np.zeros(task_count, dtype=np.bool_)
    if bool(time_scaling_cfg.get("retime_after_terminal_projection", False)):
        projection_trajectories = []
        for index, path in enumerate(q_projected):
            trajectory, timing = retime_sampled_path(
                q=path,
                predicted_duration=float(source["duration"][index]),
                min_duration=min_duration,
                nominal_max_duration=nominal_max_duration,
                max_output_duration=max_output_duration,
                velocity_limits=velocity_limits,
                acceleration_limits=acceleration_limits,
                jerk_limits=jerk_limits,
                timing_points=timing_points,
            )
            projection_trajectories.append(trajectory.q)
            projection_durations[index] = trajectory.duration
            projection_duration_valid[index] = bool(timing["duration_valid"])
            projection_duration_exceeds_nominal[index] = bool(
                timing["duration_exceeds_nominal_max"]
            )
        projection_dense = np.asarray(projection_trajectories, dtype=np.float32)
    else:
        projection_dense = np.stack(
            [periodic_resample(path, output_points) for path in q_projected]
        )
        projection_duration_valid = np.asarray(
            [
                bool(visual_records[str(sample_id)]["multi"]["duration_ok"])
                for sample_id in source["sample_id"]
            ],
            dtype=np.bool_,
        )
    projection_elapsed = perf_counter() - started
    env = TrajectoryRefinementEnv(
        source_paths=q_projected,
        csdf_model=csdf,
        config=refiner_cfg,
        device=device,
        seed=seed,
    )
    checkpoint_ppo_cfg = policy_checkpoint.get("ppo_config", {})
    policy = TrajectoryActorCritic(
        state_dim=int(policy_checkpoint["state_dim"]),
        action_dim=int(policy_checkpoint["action_dim"]),
        hidden_dim=int(checkpoint_ppo_cfg.get("hidden_dim", 384)),
        initial_std=float(checkpoint_ppo_cfg.get("initial_std", 0.35)),
    ).to(device)
    policy.load_state_dict(policy_checkpoint["model_state_dict"])
    policy.eval()
    proposal_batches = []
    refiner_batch_size = int(evaluation.get("refiner_inference_batch_size", 64))
    for batch_start in range(0, task_count, refiner_batch_size):
        batch_end = min(batch_start + refiner_batch_size, task_count)
        state, _ = env.reset(
            paths=q_projected[batch_start:batch_end], add_corruption=False
        )
        with torch.no_grad():
            for _ in range(env.episode_steps):
                action, _, _ = policy.act(state, deterministic=True)
                state, _, _, _ = env.step(action)
        proposal_batches.append(env.trajectories())
    policy_proposals = np.concatenate(proposal_batches, axis=0)
    proposal_durations = projection_durations.copy()
    proposal_duration_valid = projection_duration_valid.copy()
    proposal_duration_exceeds_nominal = projection_duration_exceeds_nominal.copy()
    if bool(time_scaling_cfg.get("retime_after_ppo", False)):
        retimed_proposals = []
        for index, path in enumerate(policy_proposals):
            trajectory, timing = retime_sampled_path(
                q=path,
                predicted_duration=float(projection_durations[index]),
                min_duration=min_duration,
                nominal_max_duration=nominal_max_duration,
                max_output_duration=max_output_duration,
                velocity_limits=velocity_limits,
                acceleration_limits=acceleration_limits,
                jerk_limits=jerk_limits,
                timing_points=timing_points,
            )
            retimed_proposals.append(trajectory.q)
            proposal_durations[index] = trajectory.duration
            proposal_duration_valid[index] = bool(timing["duration_valid"])
            proposal_duration_exceeds_nominal[index] = bool(
                timing["duration_exceeds_nominal_max"]
            )
        policy_proposals = np.asarray(retimed_proposals, dtype=np.float32)
    policy_elapsed = perf_counter() - started - projection_elapsed

    projected_reports = []
    proposal_reports = []
    final_reports = []
    final_paths = np.empty_like(policy_proposals)
    final_durations = np.empty(task_count, dtype=np.float64)
    final_duration_valid = np.empty(task_count, dtype=np.bool_)
    final_duration_exceeds_nominal = np.empty(task_count, dtype=np.bool_)
    policy_accepted = np.zeros(task_count, dtype=np.bool_)
    for index in range(task_count):
        validation_args = {
            "oracle": oracle,
            "required_clearance_m": required_clearance,
            "max_joint_step_rad": float(
                evaluation["continuous_check_max_joint_step_rad"]
            ),
            "near_boundary_extra_subdivisions": int(
                evaluation["near_boundary_extra_subdivisions"]
            ),
        }
        projected_report = adaptive_validate_trajectory(
            projection_dense[index], **validation_args
        )
        proposal_report = adaptive_validate_trajectory(
            policy_proposals[index], **validation_args
        )
        projected_reports.append(projected_report)
        proposal_reports.append(proposal_report)
        accept = bool(
            proposal_report.safe
            and proposal_duration_valid[index]
            and (
                not projected_report.safe
                or not projection_duration_valid[index]
                or proposal_report.min_clearance_m
                > projected_report.min_clearance_m + 0.001
            )
        )
        policy_accepted[index] = accept
        final_paths[index] = (
            policy_proposals[index] if accept else projection_dense[index]
        )
        final_durations[index] = (
            proposal_durations[index] if accept else projection_durations[index]
        )
        final_duration_valid[index] = (
            proposal_duration_valid[index]
            if accept
            else projection_duration_valid[index]
        )
        final_duration_exceeds_nominal[index] = (
            proposal_duration_exceeds_nominal[index]
            if accept
            else projection_duration_exceeds_nominal[index]
        )
        final_reports.append(proposal_report if accept else projected_report)
        if (index + 1) % 25 == 0 or index + 1 == task_count:
            print(f"[SELF-CHECK] {index + 1}/{task_count}", flush=True)
    exact_elapsed = perf_counter() - started - projection_elapsed - policy_elapsed

    dynamics_reference_points = int(
        evaluation.get("dynamics_reference_points", output_points)
    )
    segments = link_segments(
        fk, final_paths.reshape(-1, 6), device=device
    ).reshape(task_count, output_points, -1, 2, 3)
    with torch.no_grad():
        terminal_positions, _ = fk(
            torch.as_tensor(final_paths[:, -1], dtype=torch.float32, device=device)
        )
    terminal_errors = torch.linalg.norm(
        terminal_positions - torch.as_tensor(goals, dtype=torch.float32, device=device),
        dim=1,
    ).cpu().numpy()

    records = []
    environment_clearances = []
    for index, sample_id_value in enumerate(source["sample_id"]):
        sample_id = str(sample_id_value)
        row = test_rows[sample_id]
        cloud = row["scene_points"][row["scene_point_mask"]]
        environment_clearance = float(
            environment_clearance_batch(
                segments[index : index + 1, :, 1:],
                cloud,
                capsule_radius_m=float(evaluation["environment_capsule_radius_m"]),
            )[0]
        )
        environment_clearances.append(environment_clearance)
        dynamics = trajectory_metrics(
            final_paths[index],
            float(final_durations[index]),
            velocity_limits,
            acceleration_limits,
            jerk_limits,
            dynamics_reference_points,
        )
        original_record = visual_records[sample_id]["multi"]
        source_duration_ok = bool(original_record["duration_ok"])
        duration_ok = bool(final_duration_valid[index])
        terminal_ok = bool(
            terminal_errors[index] <= float(evaluation["position_tolerance_m"])
        )
        environment_ok = environment_clearance >= required_clearance
        self_ok = bool(final_reports[index].safe)
        success = bool(
            terminal_ok
            and environment_ok
            and self_ok
            and dynamics["dynamics_ok"]
            and duration_ok
        )
        terminal_result = terminal_results.get(index)
        records.append(
            {
                "task_index": index,
                "sample_id": sample_id,
                "side": str(source["side"][index]),
                "success": success,
                "terminal_error_m": float(terminal_errors[index]),
                "environment_clearance_m": environment_clearance,
                "self_clearance_m": float(final_reports[index].min_clearance_m),
                "terminal_ok": terminal_ok,
                "environment_ok": environment_ok,
                "self_clearance_ok": self_ok,
                "dynamics_ok": bool(dynamics["dynamics_ok"]),
                "duration_ok": duration_ok,
                "source_duration_ok": source_duration_ok,
                "source_duration_s": float(source["duration"][index]),
                "projection_duration_s": float(projection_durations[index]),
                "proposal_duration_s": float(proposal_durations[index]),
                "final_duration_s": float(final_durations[index]),
                "duration_exceeds_nominal_max": bool(
                    final_duration_exceeds_nominal[index]
                ),
                "terminal_projection_required": index in terminal_results,
                "terminal_projection_found": bool(
                    terminal_result is None or terminal_result.found
                ),
                "terminal_projection_position_error_m": (
                    None
                    if terminal_result is None
                    else float(terminal_result.position_error_m)
                ),
                "policy_proposal_accepted": bool(policy_accepted[index]),
                "adaptive_checked_configurations": int(
                    final_reports[index].checked_configurations
                ),
                **dynamics,
            }
        )
        if (index + 1) % 25 == 0 or index + 1 == task_count:
            print(f"[ENV-CHECK] {index + 1}/{task_count}", flush=True)
    environment_elapsed = (
        perf_counter() - started - projection_elapsed - policy_elapsed - exact_elapsed
    )

    successes = sum(record["success"] for record in records)
    constraint_keys = (
        "terminal_ok",
        "environment_ok",
        "self_clearance_ok",
        "dynamics_ok",
        "duration_ok",
    )
    failure_counts = {
        key: sum(not record[key] for record in records) for key in constraint_keys
    }
    report = {
        "method": f"five-view Visual CFM v5 + safe terminal projection + C-SDF-guided PPO {output_points}-point refiner",
        "evaluation_type": "fixed independent test set; final integrated method only; no comparison or ablation groups",
        "seed": seed,
        "device": device,
        "tasks": task_count,
        "input": {
            "visual_observation": "5-view RGB-D-semantic + q_start[6] + target_position_in_arm_foot0[3]",
            "cached_visual_cfm_output_shape": [task_count, 64, 6],
            "output_shape": [task_count, output_points, 6],
            "duration_shape": [task_count],
        },
        "refiner_contract": {
            "config": str(refiner_config_path),
            "state_dim": env.state_dim,
            "control_points": env.control_count,
            "action_dim": env.action_dim,
            "episode_steps": env.episode_steps,
            "output_points": output_points,
            "inference_batch_size": refiner_batch_size,
            "dynamics_reference_points": dynamics_reference_points,
            "policy_hidden_dim": int(checkpoint_ppo_cfg.get("hidden_dim", 384)),
        },
        "provenance": {
            "visual_checkpoint": str(visual_checkpoint),
            "visual_checkpoint_sha256": sha256(visual_checkpoint),
            "visual_config": str(resolve(source_cfg["visual_config"])),
            "visual_output_cache": str(source_path),
            "visual_output_cache_sha256": sha256(source_path),
            "visual_report": str(visual_report_path),
            "csdf_checkpoint": str(csdf_checkpoint_path),
            "csdf_checkpoint_sha256": sha256(csdf_checkpoint_path),
            "refiner_checkpoint": str(refiner_checkpoint_path),
            "refiner_checkpoint_sha256": sha256(refiner_checkpoint_path),
        },
        "success_criterion": {
            "terminal_position_error_m_max": float(
                evaluation["position_tolerance_m"]
            ),
            "environment_clearance_m_min": required_clearance,
            "adaptive_self_clearance_m_min": required_clearance,
            "dynamics_within_soft_limits": True,
            "duration_is_finite_and_within_output_limit": max_output_duration,
        },
        "integrated_performance": {
            "successes": successes,
            "success_rate": successes / task_count,
            "success_rate_wilson_95": wilson(successes, task_count),
            "left_success_rate": side_success_rate(records, "left"),
            "right_success_rate": side_success_rate(records, "right"),
            "constraint_pass_rates": {
                key: float(np.mean([record[key] for record in records]))
                for key in constraint_keys
            },
            "failure_counts_nonexclusive": failure_counts,
        },
        "refiner_activity": {
            "terminal_projection_requested": int(len(terminal_indices)),
            "terminal_projection_succeeded": int(
                sum(result.found for result in terminal_results.values())
            ),
            "ppo_proposals_accepted": int(policy_accepted.sum()),
            "projected_paths_safe": int(sum(r.safe for r in projected_reports)),
            "ppo_proposals_safe": int(sum(r.safe for r in proposal_reports)),
            "final_paths_safe": int(sum(r.safe for r in final_reports)),
        },
        "time_scaling": {
            "interpolation_precedes_time_scaling": True,
            "timing_points": timing_points,
            "retime_after_terminal_projection": bool(
                time_scaling_cfg.get("retime_after_terminal_projection", False)
            ),
            "retime_after_ppo": bool(
                time_scaling_cfg.get("retime_after_ppo", False)
            ),
            "nominal_max_duration_s": nominal_max_duration,
            "max_output_duration_s": max_output_duration,
            "hard_clip_applied": False,
            "final_duration_s": distribution(final_durations.tolist()),
            "final_duration_exceeds_nominal_tasks": int(
                final_duration_exceeds_nominal.sum()
            ),
            "final_duration_invalid_tasks": int((~final_duration_valid).sum()),
        },
        "trajectory_statistics_all_tasks": {
            key: distribution([float(record[key]) for record in records])
            for key in (
                "terminal_error_m",
                "environment_clearance_m",
                "self_clearance_m",
                "velocity_rms_rad_s",
                "acceleration_rms_rad_s2",
                "jerk_rms_rad_s3",
            )
        },
        "timing_s": {
            "terminal_projection": projection_elapsed,
            "ppo_batch_inference": policy_elapsed,
            "adaptive_self_collision_validation": exact_elapsed,
            "environment_and_metric_validation": environment_elapsed,
            "total": perf_counter() - started,
        },
        "scope": {
            "visual_environment_geometry": "five-camera reconstructed point cloud; no USD obstacle truth",
            "self_collision": "known arm kinematics and capsule geometry with adaptive interval densification",
            "physics": "this report is geometric/dynamic-limit evaluation; PhysX replay uses the exported 26-task bank",
        },
        "records": records,
    }
    output_cfg = cfg["outputs"]
    output_path = resolve(output_cfg["report"])
    trajectory_path = resolve(output_cfg["trajectories"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        trajectory_path,
        sample_id=source["sample_id"],
        side=source["side"],
        q_start=source["q_start"],
        goal_position=source["goal_position"],
        q=final_paths,
        duration=final_durations.astype(np.float32),
        source_duration=source["duration"],
        projection_duration=projection_durations.astype(np.float32),
        proposal_duration=proposal_durations.astype(np.float32),
        duration_valid=final_duration_valid,
        geometric_success=np.asarray([r["success"] for r in records], dtype=np.bool_),
        exact_self_clearance_m=np.asarray(
            [r["self_clearance_m"] for r in records], dtype=np.float32
        ),
        visual_environment_clearance_m=np.asarray(
            environment_clearances, dtype=np.float32
        ),
    )

    reference_bank = resolve(source_cfg["physx_reference_bank"])
    physx_path = resolve(output_cfg["physx_bank"])
    with np.load(reference_bank, allow_pickle=False) as reference:
        reference_ids = [str(item) for item in reference["sample_id"]]
    source_indices = {str(item): index for index, item in enumerate(source["sample_id"])}
    missing = [item for item in reference_ids if item not in source_indices]
    if missing and not args.max_tasks:
        raise KeyError(f"PhysX reference IDs absent from integrated outputs: {missing}")
    available_ids = [item for item in reference_ids if item in source_indices]
    selected = np.asarray(
        [source_indices[item] for item in available_ids], dtype=np.int64
    )
    np.savez_compressed(
        physx_path,
        sample_id=source["sample_id"][selected],
        side=source["side"][selected],
        q=final_paths[selected, None],
        duration=final_durations[selected, None].astype(np.float32),
        geometric_success=np.asarray(
            [records[index]["success"] for index in selected], dtype=np.bool_
        )[:, None],
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "trajectories": str(trajectory_path),
                "physx_bank": str(physx_path),
                **report["integrated_performance"],
                "refiner_activity": report["refiner_activity"],
                "timing_s": report["timing_s"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
