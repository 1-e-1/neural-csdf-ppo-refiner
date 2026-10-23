"""Compact PPO trainer for the residual trajectory-refinement MDP."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from .models import TrajectoryActorCritic
from .trajectory_env import TrajectoryRefinementEnv


@dataclass(frozen=True)
class PPOTrainingSummary:
    updates: int
    environments: int
    best_approximate_success_rate: float
    final_approximate_success_rate: float
    seed: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def train_ppo(
    env: TrajectoryRefinementEnv,
    policy: TrajectoryActorCritic,
    config: dict,
    seed: int,
    checkpoint_path,
) -> tuple[list[dict[str, float]], PPOTrainingSummary]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    environments = int(config["environments"])
    updates = int(config["updates"])
    gamma = float(config["gamma"])
    gae_lambda = float(config["gae_lambda"])
    clip_ratio = float(config["clip_ratio"])
    value_weight = float(config["value_weight"])
    entropy_weight = float(config["entropy_weight"])
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=float(config["learning_rate"]), weight_decay=1.0e-5
    )
    history = []
    best_success = -1.0
    best_clearance = -float("inf")

    for update in range(1, updates + 1):
        state, initial = env.reset(environments)
        initial_success = initial["success"].float().mean().item()
        states = []
        actions = []
        old_log_probs = []
        rewards = []
        values = []
        dones = []
        for _ in range(env.episode_steps):
            with torch.no_grad():
                action, log_prob, value = policy.act(state)
            next_state, reward, done, _ = env.step(action)
            states.append(state)
            actions.append(action)
            old_log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            dones.append(done)
            state = next_state

        advantages = []
        gae = torch.zeros(environments, device=state.device)
        next_value = torch.zeros_like(gae)
        for index in reversed(range(env.episode_steps)):
            nonterminal = (~dones[index]).float()
            delta = rewards[index] + gamma * next_value * nonterminal - values[index]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            advantages.append(gae)
            next_value = values[index]
        advantages.reverse()
        returns = [advantage + value for advantage, value in zip(advantages, values)]

        flat_state = torch.cat(states)
        flat_action = torch.cat(actions)
        flat_old_log_prob = torch.cat(old_log_probs)
        flat_advantage = torch.cat(advantages)
        flat_return = torch.cat(returns)
        flat_advantage = (flat_advantage - flat_advantage.mean()) / flat_advantage.std().clamp_min(1.0e-6)
        batch_size = len(flat_state)
        minibatch_size = min(int(config["minibatch_size"]), batch_size)
        policy_losses = []
        value_losses = []
        entropy_values = []
        for _ in range(int(config["ppo_epochs"])):
            permutation = torch.randperm(batch_size, device=state.device)
            for start in range(0, batch_size, minibatch_size):
                index = permutation[start : start + minibatch_size]
                log_prob, entropy, value = policy.evaluate_actions(
                    flat_state[index], flat_action[index]
                )
                ratio = torch.exp(log_prob - flat_old_log_prob[index])
                unclipped = ratio * flat_advantage[index]
                clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * flat_advantage[index]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = torch.mean((value - flat_return[index]) ** 2)
                entropy_mean = entropy.mean()
                loss = policy_loss + value_weight * value_loss - entropy_weight * entropy_mean
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), float(config["max_grad_norm"]))
                optimizer.step()
                policy_losses.append(float(policy_loss.detach()))
                value_losses.append(float(value_loss.detach()))
                entropy_values.append(float(entropy_mean.detach()))

        final_success = float(env.last_metrics["success"].float().mean())
        mean_clearance = float(env.last_metrics["min_clearance_m"].mean())
        record = {
            "update": update,
            "initial_success_rate": initial_success,
            "final_success_rate": final_success,
            "mean_final_min_clearance_m": mean_clearance,
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropy_values)),
        }
        history.append(record)
        if final_success > best_success or (
            final_success == best_success and mean_clearance > best_clearance
        ):
            best_success = final_success
            best_clearance = mean_clearance
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": policy.state_dict(),
                    "state_dim": policy.state_dim,
                    "action_dim": policy.action_dim,
                    "update": update,
                    "best_approximate_success_rate": best_success,
                    "best_mean_min_clearance_m": best_clearance,
                },
                checkpoint_path,
            )
        print(
            f"[PPO] update={update:04d} initial={initial_success:.3f} "
            f"final={final_success:.3f} clearance={mean_clearance:.4f}",
            flush=True,
        )

    return history, PPOTrainingSummary(
        updates=updates,
        environments=environments,
        best_approximate_success_rate=best_success,
        final_approximate_success_rate=history[-1]["final_success_rate"],
        seed=seed,
    )
