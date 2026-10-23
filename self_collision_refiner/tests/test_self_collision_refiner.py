from pathlib import Path
import sys

import numpy as np
import torch

SUBPROJECT = Path(__file__).resolve().parents[1]
H1_ROOT = SUBPROJECT.parent
WORKSPACE = H1_ROOT
sys.path.insert(0, str(SUBPROJECT))
sys.path.insert(0, str(H1_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE))

from self_collision_refiner.geometry import (
    ExactSelfCollisionOracle,
    cubic_interpolation_matrix,
    periodic_resample,
    project_terminal_with_minimum_jerk,
    refine_safe_terminal_configurations_batch,
    wrap_angles,
)
from self_collision_refiner.models import NeuralCSDF, TrajectoryActorCritic
from self_collision_refiner.trajectory_env import TrajectoryRefinementEnv
from src.robot.torch_kinematics import DifferentiableZBotFK


def refiner_config() -> dict:
    return {
        "control_points": 8,
        "output_points": 256,
        "episode_steps": 2,
        "action_step_rad": 0.1,
        "max_residual_rad": 0.45,
        "initial_corruption_rad": 0.0,
        "required_clearance_m": 0.01,
        "gamma": 0.99,
        "reward_weights": {
            "clearance": 1.0,
            "collision_fraction": 3.0,
            "deviation": 0.2,
            "acceleration": 0.08,
            "jerk": 0.02,
            "action": 0.01,
            "success": 5.0,
        },
    }


def test_wrap_angles_is_half_open_and_periodic() -> None:
    values = np.array([-3.0 * np.pi, -np.pi, np.pi, 3.0 * np.pi])
    wrapped = wrap_angles(values)
    assert np.all(wrapped >= -np.pi)
    assert np.all(wrapped < np.pi)
    np.testing.assert_allclose(wrapped, -np.pi)


def test_periodic_resample_preserves_endpoints_without_long_way_round() -> None:
    q = np.zeros((64, 6), dtype=np.float32)
    q[:, 0] = wrap_angles(np.linspace(3.0, 3.3, 64))
    output = periodic_resample(q, 256)
    assert output.shape == (256, 6)
    np.testing.assert_allclose(output[0], q[0], atol=1.0e-6)
    np.testing.assert_allclose(output[-1], q[-1], atol=1.0e-6)
    unwrapped = np.unwrap(output[:, 0], period=2.0 * np.pi)
    assert np.max(np.abs(np.diff(unwrapped))) < 0.02


def test_cubic_basis_preserves_first_and_last_controls() -> None:
    basis = cubic_interpolation_matrix(8, 256)
    np.testing.assert_array_equal(basis[0], np.eye(8, dtype=np.float32)[0])
    np.testing.assert_array_equal(basis[-1], np.eye(8, dtype=np.float32)[-1])


def test_terminal_projection_keeps_start_and_sets_periodic_terminal() -> None:
    q = np.zeros((64, 6), dtype=np.float32)
    q[:, 0] = np.linspace(0.0, 0.4, 64)
    terminal = np.array([0.6, -0.2, 0.1, 0.0, 0.0, 0.0], dtype=np.float32)
    projected = project_terminal_with_minimum_jerk(q, terminal)
    np.testing.assert_allclose(projected[0], q[0], atol=1.0e-7)
    np.testing.assert_allclose(projected[-1], terminal, atol=1.0e-6)


def test_csdf_and_policy_shapes() -> None:
    csdf = NeuralCSDF((32, 32))
    clearance, logit = csdf(torch.zeros(5, 6))
    assert clearance.shape == (5,)
    assert logit.shape == (5,)
    policy = TrajectoryActorCritic(20, 12, hidden_dim=32)
    action, log_prob, value = policy.act(torch.zeros(4, 20))
    assert action.shape == (4, 12)
    assert log_prob.shape == (4,)
    assert value.shape == (4,)


def test_environment_exports_256_points_and_fixed_endpoints() -> None:
    source = np.zeros((3, 64, 6), dtype=np.float32)
    source[:, :, 0] = np.linspace(-0.2, 0.4, 64)
    env = TrajectoryRefinementEnv(source, NeuralCSDF((32, 32)), refiner_config(), "cpu", 7)
    state, _ = env.reset(paths=source[:2], add_corruption=False)
    assert state.shape == (2, env.state_dim)
    action = torch.zeros(2, env.action_dim)
    env.step(action)
    output = env.trajectories()
    assert output.shape == (2, 256, 6)
    np.testing.assert_allclose(output[:, 0], source[:2, 0], atol=1.0e-6)
    np.testing.assert_allclose(output[:, -1], source[:2, -1], atol=1.0e-6)


def test_full64_environment_contract() -> None:
    source = np.zeros((2, 64, 6), dtype=np.float32)
    config = refiner_config()
    config["control_points"] = 64
    env = TrajectoryRefinementEnv(source, NeuralCSDF((32, 32)), config, "cpu", 9)
    state, _ = env.reset(paths=source, add_corruption=False)
    assert env.control_count == 64
    assert env.action_dim == 62
    assert env.state_dim == 1976
    assert state.shape == (2, 1976)
    _, _, _, _ = env.step(torch.zeros(2, 62))
    assert env.trajectories().shape == (2, 256, 6)


def test_full64_environment_exports_configured_1024_points() -> None:
    source = np.zeros((2, 64, 6), dtype=np.float32)
    source[:, :, 0] = np.linspace(-0.2, 0.4, 64)
    config = refiner_config()
    config["control_points"] = 64
    config["output_points"] = 1024
    config["derivative_reference_points"] = 256
    env = TrajectoryRefinementEnv(source, NeuralCSDF((32, 32)), config, "cpu", 10)
    env.reset(paths=source, add_corruption=False)
    env.step(torch.zeros(2, 62))
    output = env.trajectories()
    assert output.shape == (2, 1024, 6)
    np.testing.assert_allclose(output[:, 0], source[:, 0], atol=1.0e-6)
    np.testing.assert_allclose(output[:, -1], source[:, -1], atol=1.0e-6)


def test_derivative_proxy_is_resolution_invariant() -> None:
    source = np.zeros((1, 64, 6), dtype=np.float32)
    phase = np.linspace(0.0, 1.0, 64)
    source[0, :, 0] = 0.3 * np.sin(np.pi * phase)
    metrics = []
    for output_points in (256, 1024):
        config = refiner_config()
        config["control_points"] = 64
        config["output_points"] = output_points
        config["derivative_reference_points"] = 256
        env = TrajectoryRefinementEnv(source, NeuralCSDF((32, 32)), config, "cpu", 12)
        _, measured = env.reset(paths=source, add_corruption=False)
        metrics.append(measured)
    np.testing.assert_allclose(
        float(metrics[0]["acceleration_proxy"][0]),
        float(metrics[1]["acceleration_proxy"][0]),
        rtol=0.03,
    )
    np.testing.assert_allclose(
        float(metrics[0]["jerk_proxy"][0]),
        float(metrics[1]["jerk_proxy"][0]),
        rtol=0.08,
    )


def test_smooth_reward_exposes_adjacent_residual_oscillation() -> None:
    source = np.zeros((1, 64, 6), dtype=np.float32)
    config = refiner_config()
    config["control_points"] = 64
    config["proxy_scales"] = {
        "acceleration": 1.0e-5,
        "jerk": 5.0e-6,
        "oscillation": 5.0e-5,
    }
    config["reward_weights"]["oscillation"] = 1.0
    env = TrajectoryRefinementEnv(source, NeuralCSDF((32, 32)), config, "cpu", 11)
    env.reset(paths=source, add_corruption=False)
    alternating = torch.tensor([[1.0, -1.0] * 31])
    _, reward, _, metrics = env.step(alternating)
    assert float(metrics["oscillation_proxy"][0]) > 0.0
    assert torch.isfinite(reward).all()


def test_exact_oracle_returns_one_clearance_per_configuration() -> None:
    fk = DifferentiableZBotFK(WORKSPACE / "configs" / "kinematic_chain.json")
    oracle = ExactSelfCollisionOracle(fk, device="cpu", batch_size=4)
    clearance = oracle.clearance(np.zeros((5, 6), dtype=np.float32))
    assert clearance.shape == (5,)
    assert np.isfinite(clearance).all()


def test_batched_terminal_search_returns_one_result_per_target() -> None:
    fk = DifferentiableZBotFK(WORKSPACE / "configs" / "kinematic_chain.json")
    oracle = ExactSelfCollisionOracle(fk, device="cpu", batch_size=4)
    q = np.zeros((2, 6), dtype=np.float32)
    with torch.no_grad():
        targets, _ = fk(torch.as_tensor(q))
    results = refine_safe_terminal_configurations_batch(
        fk_model=fk,
        csdf_model=NeuralCSDF((16, 16)),
        oracle=oracle,
        q_initial=q,
        target_positions=targets.numpy(),
        required_clearance_m=0.01,
        device="cpu",
        restarts=2,
        iterations=1,
        seed=7,
    )
    assert len(results) == 2
    assert all(result.q.shape == (6,) for result in results)
    assert all(np.isfinite(result.exact_clearance_m) for result in results)
