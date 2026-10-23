"""Train the larger full-64-point PPO self-collision refiner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import yaml

import _bootstrap

from self_collision_refiner.models import NeuralCSDF, TrajectoryActorCritic
from self_collision_refiner.ppo import train_ppo
from self_collision_refiner.trajectory_env import TrajectoryRefinementEnv


def resolve(value: str) -> Path:
    return (_bootstrap.FULL64_ROOT / value).resolve()


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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train_path = resolve(cfg["task_generation"]["train_output"])
    with np.load(train_path, allow_pickle=False) as data:
        source_paths = np.asarray(data["q"], dtype=np.float32)
    csdf_path = resolve(cfg["models"]["csdf_checkpoint"])
    csdf_checkpoint = torch.load(csdf_path, map_location=device, weights_only=False)
    csdf = NeuralCSDF(tuple(csdf_checkpoint["hidden_dims"])).to(device)
    csdf.load_state_dict(csdf_checkpoint["model_state_dict"])
    csdf.eval()
    refiner_cfg = dict(cfg["refiner"])
    ppo_cfg = dict(cfg["ppo"])
    env = TrajectoryRefinementEnv(
        source_paths=source_paths,
        csdf_model=csdf,
        config=refiner_cfg,
        device=device,
        seed=seed,
    )
    if env.control_count != 64 or env.action_dim != 62:
        raise RuntimeError(
            f"full64 contract violated: controls={env.control_count}, actions={env.action_dim}"
        )
    policy = TrajectoryActorCritic(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=int(ppo_cfg["hidden_dim"]),
        initial_std=float(ppo_cfg["initial_std"]),
    ).to(device)
    initialization_checkpoint = ppo_cfg.get("initialize_from")
    if initialization_checkpoint:
        initialization_path = resolve(initialization_checkpoint)
        initialized = torch.load(
            initialization_path, map_location=device, weights_only=False
        )
        if int(initialized["state_dim"]) != env.state_dim:
            raise ValueError("initialization checkpoint observation dimension mismatch")
        if int(initialized["action_dim"]) != env.action_dim:
            raise ValueError("initialization checkpoint action dimension mismatch")
        policy.load_state_dict(initialized["model_state_dict"])
    else:
        initialization_path = None
    parameters = sum(parameter.numel() for parameter in policy.parameters())
    checkpoint = resolve(ppo_cfg["checkpoint"])
    started = perf_counter()
    history, summary = train_ppo(
        env=env,
        policy=policy,
        config=ppo_cfg,
        seed=seed,
        checkpoint_path=checkpoint,
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved.update(
        {
            "refiner_config": refiner_cfg,
            "ppo_config": ppo_cfg,
            "csdf_checkpoint": str(csdf_path),
            "training_tasks": str(train_path),
            "parameter_count": parameters,
            "architecture": "full 64 controls, 62 C-SDF gradient-gate actions",
        }
    )
    torch.save(saved, checkpoint)
    report = {
        **summary.to_dict(),
        "elapsed_s": perf_counter() - started,
        "checkpoint": str(checkpoint),
        "csdf_checkpoint": str(csdf_path),
        "training_tasks": len(source_paths),
        "observation_dim": env.state_dim,
        "action_dim": env.action_dim,
        "control_points": env.control_count,
        "hidden_dim": int(ppo_cfg["hidden_dim"]),
        "parameter_count": parameters,
        "input_shape": [64, 6],
        "output_shape": [env.output_points, 6],
        "initialization_checkpoint": (
            None if initialization_path is None else str(initialization_path)
        ),
        "history": history,
    }
    report_path = resolve(ppo_cfg["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "history"},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
