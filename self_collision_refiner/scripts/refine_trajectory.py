"""Run PPO self-collision refinement and export a 256x6 trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

import _bootstrap

from self_collision_refiner.geometry import (
    ExactSelfCollisionOracle,
    adaptive_validate_trajectory,
    periodic_resample,
    project_terminal_with_minimum_jerk,
    refine_safe_terminal_configuration,
)
from self_collision_refiner.models import NeuralCSDF, TrajectoryActorCritic
from self_collision_refiner.trajectory_env import TrajectoryRefinementEnv
from src.robot.torch_kinematics import DifferentiableZBotFK


def resolve(value: str) -> Path:
    return (_bootstrap.SUBPROJECT_ROOT / value).resolve()


def read_path(path: Path, index: int) -> tuple[np.ndarray, float, str]:
    with np.load(path, allow_pickle=False) as data:
        q = np.asarray(data["q"], dtype=np.float32)
        if q.ndim == 3:
            q = q[index]
        if q.shape != (64, 6):
            raise ValueError(f"input q must resolve to [64, 6], got {q.shape}")
        duration = float(np.asarray(data["duration"]).reshape(-1)[index]) if "duration" in data else 4.0
        sample_id = str(np.asarray(data["sample_id"]).reshape(-1)[index]) if "sample_id" in data else str(index)
    return q, duration, sample_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=_bootstrap.SUBPROJECT_ROOT / "config.yaml")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = args.device or cfg["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    input_path = args.input or resolve(cfg["inference"]["input"])
    output_path = args.output or resolve(cfg["inference"]["output"])
    sample_index = args.index if args.index is not None else int(cfg["inference"]["sample_index"])
    q64, duration, sample_id = read_path(input_path, sample_index)

    csdf_checkpoint = torch.load(
        resolve(cfg["csdf_training"]["checkpoint"]), map_location=device, weights_only=False
    )
    csdf = NeuralCSDF(tuple(csdf_checkpoint["hidden_dims"])).to(device)
    csdf.load_state_dict(csdf_checkpoint["model_state_dict"])
    csdf.eval()
    geometry = cfg["geometry"]
    fk = DifferentiableZBotFK(resolve(geometry["kinematic_chain"])).to(device)
    oracle = ExactSelfCollisionOracle(
        fk,
        capsule_radius_m=geometry["capsule_radius_m"],
        adjacency=geometry["adjacency"],
        device=device,
        batch_size=geometry["exact_query_batch_size"],
    )
    original_report = adaptive_validate_trajectory(
        periodic_resample(q64, 256),
        oracle,
        required_clearance_m=geometry["required_clearance_m"],
        max_joint_step_rad=geometry["continuous_check_max_joint_step_rad"],
        near_boundary_extra_subdivisions=geometry["near_boundary_extra_subdivisions"],
    )
    endpoint_clearance = oracle.clearance(q64[[0, -1]])
    with torch.no_grad():
        target_position, _ = fk(torch.as_tensor(q64[-1], dtype=torch.float32, device=device))
    q_policy = q64.copy()
    terminal_result = None
    if endpoint_clearance[0] >= geometry["required_clearance_m"] and endpoint_clearance[1] < geometry["required_clearance_m"]:
        terminal_result = refine_safe_terminal_configuration(
            fk_model=fk,
            csdf_model=csdf,
            oracle=oracle,
            q_initial=q64[-1],
            required_clearance_m=float(geometry["required_clearance_m"]),
            device=device,
            seed=int(cfg["seed"]) + sample_index,
        )
        if terminal_result.found:
            q_policy = project_terminal_with_minimum_jerk(q64, terminal_result.q)
    projected_report = adaptive_validate_trajectory(
        periodic_resample(q_policy, 256),
        oracle,
        required_clearance_m=geometry["required_clearance_m"],
        max_joint_step_rad=geometry["continuous_check_max_joint_step_rad"],
        near_boundary_extra_subdivisions=geometry["near_boundary_extra_subdivisions"],
    )
    policy_checkpoint = torch.load(
        resolve(cfg["ppo"]["checkpoint"]), map_location=device, weights_only=False
    )
    env = TrajectoryRefinementEnv(
        source_paths=q_policy[None],
        csdf_model=csdf,
        config=cfg["refiner"],
        device=device,
        seed=int(cfg["seed"]),
    )
    policy = TrajectoryActorCritic(
        state_dim=int(policy_checkpoint["state_dim"]),
        action_dim=int(policy_checkpoint["action_dim"]),
        hidden_dim=int(cfg["ppo"]["hidden_dim"]),
        initial_std=float(cfg["ppo"]["initial_std"]),
    ).to(device)
    policy.load_state_dict(policy_checkpoint["model_state_dict"])
    policy.eval()
    state, initial_metrics = env.reset(paths=q_policy[None], add_corruption=False)
    with torch.no_grad():
        for _ in range(env.episode_steps):
            action, _, _ = policy.act(state, deterministic=True)
            state, _, _, final_metrics = env.step(action)
    refined = env.trajectories()[0]

    refined_report = adaptive_validate_trajectory(
        refined,
        oracle,
        required_clearance_m=geometry["required_clearance_m"],
        max_joint_step_rad=geometry["continuous_check_max_joint_step_rad"],
        near_boundary_extra_subdivisions=geometry["near_boundary_extra_subdivisions"],
    )
    with torch.no_grad():
        refined_terminal_position, _ = fk(
            torch.as_tensor(refined[-1], dtype=torch.float32, device=device)
        )
    target_position_error = float(torch.linalg.norm(refined_terminal_position - target_position))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        sample_id=np.asarray(sample_id),
        q_original=q64,
        q_terminal_projected=q_policy,
        q=refined,
        time=np.linspace(0.0, duration, 256, dtype=np.float32),
        duration=np.float32(duration),
        exact_self_collision_safe=np.bool_(refined_report.safe),
        exact_min_clearance_m=np.float32(refined_report.min_clearance_m),
    )
    report = {
        "sample_id": sample_id,
        "input": str(input_path),
        "input_shape": [64, 6],
        "output": str(output_path),
        "output_shape": [256, 6],
        "start_joint_state_preserved": bool(np.allclose(refined[0], q64[0], atol=1.0e-5)),
        "cartesian_target_preserved": target_position_error <= 0.003,
        "target_position_error_m": target_position_error,
        "endpoint_clearance_m": {
            "start": float(endpoint_clearance[0]),
            "original_terminal": float(endpoint_clearance[1]),
            "projected_terminal": float(oracle.clearance(refined[[-1]])[0]),
        },
        "endpoint_feasibility": (
            "start_infeasible"
            if endpoint_clearance[0] < geometry["required_clearance_m"]
            else "terminal_projection_failed"
            if terminal_result is not None and not terminal_result.found
            else "feasible"
        ),
        "terminal_projection": None
        if terminal_result is None
        else {
            "found": terminal_result.found,
            "position_error_m": terminal_result.position_error_m,
            "predicted_clearance_m": terminal_result.predicted_clearance_m,
            "exact_clearance_m": terminal_result.exact_clearance_m,
            "restart_index": terminal_result.restart_index,
        },
        "neural_initial_min_clearance_m": float(initial_metrics["min_clearance_m"][0]),
        "neural_final_min_clearance_m": float(final_metrics["min_clearance_m"][0]),
        "exact_uniform_256": original_report.to_dict(),
        "exact_terminal_projected_256": projected_report.to_dict(),
        "exact_rl_refined_256": refined_report.to_dict(),
        "scope": "self-collision only; environment and inter-robot collision are not evaluated",
    }
    report_path = args.report or resolve(cfg["inference"]["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
