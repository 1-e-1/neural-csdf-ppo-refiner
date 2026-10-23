"""Generate the periodic six-dimensional self-collision C-space dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import yaml

import _bootstrap

from self_collision_refiner.csdf_data import generate_csdf_dataset
from self_collision_refiner.geometry import ExactSelfCollisionOracle
from src.robot.torch_kinematics import DifferentiableZBotFK


def resolve(base: Path, value: str) -> Path:
    return (base / value).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=_bootstrap.SUBPROJECT_ROOT / "config.yaml")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    geometry = cfg["geometry"]
    dataset_cfg = cfg["csdf_dataset"]
    seed = int(cfg["seed"])
    device = args.device or cfg["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    fk = DifferentiableZBotFK(resolve(_bootstrap.SUBPROJECT_ROOT, geometry["kinematic_chain"])).to(device)
    oracle = ExactSelfCollisionOracle(
        fk,
        capsule_radius_m=geometry["capsule_radius_m"],
        adjacency=geometry["adjacency"],
        device=device,
        batch_size=geometry["exact_query_batch_size"],
    )
    started = perf_counter()
    payload, summary = generate_csdf_dataset(
        oracle=oracle,
        count=args.samples or int(dataset_cfg["samples"]),
        seed=seed,
        collision_margin_m=float(dataset_cfg["collision_margin_m"]),
        boundary_width_m=float(dataset_cfg["boundary_width_m"]),
        boundary_fraction=float(dataset_cfg["boundary_fraction"]),
        perturbation_std_rad=float(dataset_cfg["perturbation_std_rad"]),
        target_safe_fraction=float(dataset_cfg.get("target_safe_fraction", 0.25)),
    )
    output = resolve(_bootstrap.SUBPROJECT_ROOT, dataset_cfg["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    report = {
        **summary.to_dict(),
        "elapsed_s": perf_counter() - started,
        "output": str(output),
        "joint_domain_rad": [-float(np.pi), float(np.pi)],
        "label": "minimum non-adjacent capsule self-clearance; environment collisions excluded",
        "capsule_radius_m": float(geometry["capsule_radius_m"]),
        "adjacency": int(geometry["adjacency"]),
    }
    report_path = resolve(_bootstrap.SUBPROJECT_ROOT, dataset_cfg["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
