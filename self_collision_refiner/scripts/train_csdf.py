"""Train the neural configuration-space self-collision distance field."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import yaml

import _bootstrap

from self_collision_refiner.csdf_training import train_csdf
from self_collision_refiner.models import NeuralCSDF


def resolve(value: str) -> Path:
    return (_bootstrap.SUBPROJECT_ROOT / value).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=_bootstrap.SUBPROJECT_ROOT / "config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    train_cfg = dict(cfg["csdf_training"])
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    device = args.device or cfg["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    with np.load(resolve(cfg["csdf_dataset"]["output"]), allow_pickle=False) as data:
        dataset = {key: data[key] for key in data.files}
    model = NeuralCSDF(tuple(train_cfg["hidden_dims"])).to(device)
    checkpoint = resolve(train_cfg["checkpoint"])
    started = perf_counter()
    history, summary = train_csdf(
        dataset=dataset,
        model=model,
        config=train_cfg,
        margin_m=float(cfg["csdf_dataset"]["collision_margin_m"]),
        device=device,
        seed=int(cfg["seed"]),
        checkpoint_path=checkpoint,
    )
    report = {
        **summary.to_dict(),
        "elapsed_s": perf_counter() - started,
        "checkpoint": str(checkpoint),
        "history": history,
    }
    report_path = resolve(train_cfg["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
