"""Differentiable forward kinematics generated from USD joint frames."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat((quat[..., :1], -quat[..., 1:]), dim=-1)


def quat_apply(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    q_vector = quat[..., 1:]
    uv = torch.cross(q_vector, vector, dim=-1)
    uuv = torch.cross(q_vector, uv, dim=-1)
    return vector + 2.0 * (quat[..., :1] * uv + uuv)


def compose_pose(
    position_a: torch.Tensor,
    quaternion_a: torch.Tensor,
    position_b: torch.Tensor,
    quaternion_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return position_a + quat_apply(quaternion_a, position_b), quat_mul(quaternion_a, quaternion_b)


class DifferentiableZBotFK(nn.Module):
    def __init__(self, chain_config: str | Path) -> None:
        super().__init__()
        with Path(chain_config).open("r", encoding="utf-8") as stream:
            config = json.load(stream)
        self.joint_names = list(config["joint_names"])
        chain = config["chain"]
        self.chain_body_names = [str(item["body_to"]).rsplit("/", 1)[-1] for item in chain]
        axis_vectors = {"X": [1.0, 0.0, 0.0], "Y": [0.0, 1.0, 0.0], "Z": [0.0, 0.0, 1.0], None: [0.0, 0.0, 0.0]}
        self.register_buffer("local_pos_from", torch.tensor([item["local_pos_from"] for item in chain], dtype=torch.float32))
        self.register_buffer("local_quat_from", torch.tensor([item["local_quat_from_wxyz"] for item in chain], dtype=torch.float32))
        self.register_buffer("local_pos_to", torch.tensor([item["local_pos_to"] for item in chain], dtype=torch.float32))
        self.register_buffer("local_quat_to", torch.tensor([item["local_quat_to_wxyz"] for item in chain], dtype=torch.float32))
        self.register_buffer("axes", torch.tensor([axis_vectors[item["axis"]] for item in chain], dtype=torch.float32))
        self.register_buffer("angle_sign", torch.tensor([item["angle_sign"] for item in chain], dtype=torch.float32))
        self.register_buffer("joint_indices", torch.tensor([item["controlled_index"] for item in chain], dtype=torch.long))
        self.register_buffer("tcp_position", torch.tensor(config["tcp_local_position"], dtype=torch.float32))
        self.register_buffer("tcp_quaternion", torch.tensor(config["tcp_local_quaternion_wxyz"], dtype=torch.float32))

    def forward(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = q.shape[:-1]
        if q.shape[-1] != 6:
            raise ValueError(f"Expected q[..., 6], got {tuple(q.shape)}")
        q_flat = q.reshape(-1, 6)
        batch = q_flat.shape[0]
        position = torch.zeros((batch, 3), dtype=q.dtype, device=q.device)
        quaternion = torch.zeros((batch, 4), dtype=q.dtype, device=q.device)
        quaternion[:, 0] = 1.0
        for index in range(self.joint_indices.numel()):
            pos_from = self.local_pos_from[index].to(dtype=q.dtype).expand(batch, -1)
            quat_from = self.local_quat_from[index].to(dtype=q.dtype).expand(batch, -1)
            position, quaternion = compose_pose(position, quaternion, pos_from, quat_from)

            joint_index = int(self.joint_indices[index].item())
            if joint_index >= 0:
                angle = q_flat[:, joint_index] * self.angle_sign[index].to(dtype=q.dtype)
                half_angle = 0.5 * angle
                axis = self.axes[index].to(dtype=q.dtype).expand(batch, -1)
                rotation = torch.cat((torch.cos(half_angle)[:, None], axis * torch.sin(half_angle)[:, None]), dim=-1)
                zero = torch.zeros((batch, 3), dtype=q.dtype, device=q.device)
                position, quaternion = compose_pose(position, quaternion, zero, rotation)

            quat_to_inverse = quat_conjugate(self.local_quat_to[index].to(dtype=q.dtype)).expand(batch, -1)
            pos_to_inverse = -quat_apply(
                quat_to_inverse,
                self.local_pos_to[index].to(dtype=q.dtype).expand(batch, -1),
            )
            position, quaternion = compose_pose(position, quaternion, pos_to_inverse, quat_to_inverse)

        tcp_pos = self.tcp_position.to(dtype=q.dtype).expand(batch, -1)
        tcp_quat = self.tcp_quaternion.to(dtype=q.dtype).expand(batch, -1)
        position, quaternion = compose_pose(position, quaternion, tcp_pos, tcp_quat)
        quaternion = quaternion / torch.linalg.norm(quaternion, dim=-1, keepdim=True).clamp_min(1.0e-12)
        quaternion = torch.where(quaternion[:, :1] < 0.0, -quaternion, quaternion)
        return position.reshape(*original_shape, 3), quaternion.reshape(*original_shape, 4)

    def link_poses(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return every destination body frame in chain order."""
        original_shape = q.shape[:-1]
        if q.shape[-1] != 6:
            raise ValueError(f"Expected q[..., 6], got {tuple(q.shape)}")
        q_flat = q.reshape(-1, 6)
        batch = q_flat.shape[0]
        position = torch.zeros((batch, 3), dtype=q.dtype, device=q.device)
        quaternion = torch.zeros((batch, 4), dtype=q.dtype, device=q.device)
        quaternion[:, 0] = 1.0
        positions = []
        quaternions = []
        for index in range(self.joint_indices.numel()):
            pos_from = self.local_pos_from[index].to(dtype=q.dtype).expand(batch, -1)
            quat_from = self.local_quat_from[index].to(dtype=q.dtype).expand(batch, -1)
            position, quaternion = compose_pose(position, quaternion, pos_from, quat_from)

            joint_index = int(self.joint_indices[index].item())
            if joint_index >= 0:
                angle = q_flat[:, joint_index] * self.angle_sign[index].to(dtype=q.dtype)
                half_angle = 0.5 * angle
                axis = self.axes[index].to(dtype=q.dtype).expand(batch, -1)
                rotation = torch.cat(
                    (torch.cos(half_angle)[:, None], axis * torch.sin(half_angle)[:, None]),
                    dim=-1,
                )
                zero = torch.zeros((batch, 3), dtype=q.dtype, device=q.device)
                position, quaternion = compose_pose(position, quaternion, zero, rotation)

            quat_to_inverse = quat_conjugate(self.local_quat_to[index].to(dtype=q.dtype)).expand(batch, -1)
            pos_to_inverse = -quat_apply(
                quat_to_inverse,
                self.local_pos_to[index].to(dtype=q.dtype).expand(batch, -1),
            )
            position, quaternion = compose_pose(position, quaternion, pos_to_inverse, quat_to_inverse)
            quaternion = quaternion / torch.linalg.norm(quaternion, dim=-1, keepdim=True).clamp_min(1.0e-12)
            positions.append(position)
            quaternions.append(quaternion)
        return (
            torch.stack(positions, dim=-2).reshape(*original_shape, len(positions), 3),
            torch.stack(quaternions, dim=-2).reshape(*original_shape, len(quaternions), 4),
        )
