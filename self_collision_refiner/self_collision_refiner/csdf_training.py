"""Training utilities for the neural configuration-space distance field."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .models import NeuralCSDF


@dataclass(frozen=True)
class CSDFTrainingSummary:
    epochs: int
    train_samples: int
    validation_samples: int
    best_validation_loss: float
    best_validation_mae_m: float
    best_collision_accuracy: float
    best_collision_recall: float
    seed: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def train_csdf(
    dataset: dict[str, np.ndarray],
    model: NeuralCSDF,
    config: dict,
    margin_m: float,
    device: str,
    seed: int,
    checkpoint_path,
) -> tuple[list[dict[str, float]], CSDFTrainingSummary]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    count = len(dataset["q"])
    permutation = rng.permutation(count)
    validation_count = max(1, int(round(count * float(config["validation_fraction"]))))
    validation_indices = permutation[:validation_count]
    train_indices = permutation[validation_count:]

    q = torch.as_tensor(dataset["q"], dtype=torch.float32)
    clearance = torch.as_tensor(dataset["clearance_m"], dtype=torch.float32)
    collision = torch.as_tensor(dataset["collision"], dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(q[train_indices], clearance[train_indices], collision[train_indices]),
        batch_size=int(config["batch_size"]),
        shuffle=True,
    )
    validation_loader = DataLoader(
        TensorDataset(q[validation_indices], clearance[validation_indices], collision[validation_indices]),
        batch_size=int(config["batch_size"]),
        shuffle=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    positive = float(collision[train_indices].sum())
    negative = len(train_indices) - positive
    positive_weight = torch.tensor(
        negative / max(positive, 1.0), dtype=torch.float32, device=device
    )
    best_loss = float("inf")
    best_metrics = {}
    history = []

    for epoch in range(1, int(config["epochs"]) + 1):
        epoch_values = {}
        for split, loader in (("train", train_loader), ("validation", validation_loader)):
            model.train(split == "train")
            totals = {"loss": 0.0, "mae": 0.0, "correct": 0.0, "tp": 0.0, "fn": 0.0}
            samples = 0
            for q_batch, clearance_batch, collision_batch in loader:
                q_batch = q_batch.to(device)
                clearance_batch = clearance_batch.to(device)
                collision_batch = collision_batch.to(device)
                if split == "train":
                    optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(split == "train"):
                    predicted, logit = model(q_batch)
                    boundary_weight = 1.0 + float(config["boundary_weight_gain"]) * torch.exp(
                        -torch.abs(clearance_batch - margin_m)
                        / float(config["boundary_temperature_m"])
                    )
                    clearance_scale = float(config.get("clearance_scale_m", 0.10))
                    regression = F.smooth_l1_loss(
                        predicted / clearance_scale,
                        clearance_batch / clearance_scale,
                        beta=0.10,
                        reduction="none",
                    )
                    regression = torch.mean(boundary_weight * regression)
                    classification = F.binary_cross_entropy_with_logits(
                        logit, collision_batch, pos_weight=positive_weight
                    )
                    loss = (
                        float(config["clearance_loss_weight"]) * regression
                        + float(config["collision_loss_weight"]) * classification
                    )
                if split == "train":
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                predicted_collision = logit >= 0.0
                target_collision = collision_batch.bool()
                batch_count = len(q_batch)
                totals["loss"] += float(loss.detach()) * batch_count
                totals["mae"] += float(torch.abs(predicted - clearance_batch).sum())
                totals["correct"] += float((predicted_collision == target_collision).sum())
                totals["tp"] += float((predicted_collision & target_collision).sum())
                totals["fn"] += float((~predicted_collision & target_collision).sum())
                samples += batch_count
            epoch_values[f"{split}_loss"] = totals["loss"] / samples
            epoch_values[f"{split}_mae_m"] = totals["mae"] / samples
            epoch_values[f"{split}_collision_accuracy"] = totals["correct"] / samples
            epoch_values[f"{split}_collision_recall"] = totals["tp"] / max(
                totals["tp"] + totals["fn"], 1.0
            )
        history.append({"epoch": epoch, **epoch_values})
        if epoch_values["validation_loss"] < best_loss:
            best_loss = epoch_values["validation_loss"]
            best_metrics = dict(epoch_values)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "hidden_dims": list(config["hidden_dims"]),
                    "collision_margin_m": margin_m,
                    "epoch": epoch,
                    "best_validation_loss": best_loss,
                },
                checkpoint_path,
            )
        print(
            f"[C-SDF] epoch={epoch:03d} train={epoch_values['train_loss']:.6f} "
            f"validation={epoch_values['validation_loss']:.6f} "
            f"mae={epoch_values['validation_mae_m']:.5f}",
            flush=True,
        )

    summary = CSDFTrainingSummary(
        epochs=int(config["epochs"]),
        train_samples=len(train_indices),
        validation_samples=len(validation_indices),
        best_validation_loss=best_loss,
        best_validation_mae_m=float(best_metrics["validation_mae_m"]),
        best_collision_accuracy=float(best_metrics["validation_collision_accuracy"]),
        best_collision_recall=float(best_metrics["validation_collision_recall"]),
        seed=seed,
    )
    return history, summary
