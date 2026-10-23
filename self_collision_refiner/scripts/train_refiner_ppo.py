"""Train the PPO residual policy that refines 64 points into 256 points."""

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
    return (_bootstrap.SUBPROJECT_ROOT / value).resolve()


def load_source_paths(directory: Path, maximum: int) -> np.ndarray:
    paths = []
    for shard_path in sorted(directory.glob("train_shard_*.npz")):
        with np.load(shard_path, allow_pickle=False) as data:
            paths.append(np.asarray(data["q"], dtype=np.float32))
        if sum(len(item) for item in paths) >= maximum:
            break
    if not paths:
        raise FileNotFoundError(f"no train_shard_*.npz under {directory}")
    return np.concatenate(paths, axis=0)[:maximum]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=_bootstrap.SUBPROJECT_ROOT / "config.yaml")
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--environments", type=int, default=None)
    parser.add_argument("--max_source_trajectories", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ppo_cfg = dict(cfg["ppo"])
    refiner_cfg = dict(cfg["refiner"])
    if args.updates is not None:
        ppo_cfg["updates"] = args.updates
    if args.environments is not None:
        ppo_cfg["environments"] = args.environments
    if args.max_source_trajectories is not None:
        refiner_cfg["max_source_trajectories"] = args.max_source_trajectories
    device = args.device or cfg["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    csdf_checkpoint = torch.load(
        resolve(cfg["csdf_training"]["checkpoint"]), map_location=device, weights_only=False
    )
    csdf = NeuralCSDF(tuple(csdf_checkpoint["hidden_dims"])).to(device)
    csdf.load_state_dict(csdf_checkpoint["model_state_dict"])
    csdf.eval()
    source_paths = load_source_paths(
        resolve(refiner_cfg["source_dataset"]), int(refiner_cfg["max_source_trajectories"])
    )
    env = TrajectoryRefinementEnv(
        source_paths=source_paths,
        csdf_model=csdf,
        config=refiner_cfg,
        device=device,
        seed=int(cfg["seed"]),
    )
    policy = TrajectoryActorCritic(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=int(ppo_cfg["hidden_dim"]),
        initial_std=float(ppo_cfg["initial_std"]),
    ).to(device)
    checkpoint = resolve(ppo_cfg["checkpoint"])
    started = perf_counter()
    history, summary = train_ppo(
        env=env,
        policy=policy,
        config=ppo_cfg,
        seed=int(cfg["seed"]),
        checkpoint_path=checkpoint,
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved["refiner_config"] = refiner_cfg
    saved["ppo_config"] = ppo_cfg
    saved["csdf_checkpoint"] = str(resolve(cfg["csdf_training"]["checkpoint"]))
    saved["source_dataset"] = str(resolve(refiner_cfg["source_dataset"]))
    torch.save(saved, checkpoint)
    report = {
        **summary.to_dict(),
        "elapsed_s": perf_counter() - started,
        "checkpoint": str(checkpoint),
        "csdf_checkpoint": str(resolve(cfg["csdf_training"]["checkpoint"])),
        "source_trajectories": len(source_paths),
        "observation_dim": env.state_dim,
        "action_dim": env.action_dim,
        "input_shape": [64, 6],
        "output_shape": [256, 6],
        "history": history,
    }
    report_path = resolve(ppo_cfg["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
