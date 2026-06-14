from pathlib import Path
import shutil
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mathbased_mcpp.config import CUAPConfig, GridCoverageConfig, load_config
from mathbased_mcpp.cuap import build_bounded_prior, build_cuap_step_inputs, compute_cuap_logits
from mathbased_mcpp.env import GridCoverageEnv
from mathbased_mcpp.ppo import ActorCritic
from mathbased_mcpp.training import train_ppo


class CuapTests(unittest.TestCase):
    def test_cuap_shape_and_no_nan(self) -> None:
        env = GridCoverageEnv(
            GridCoverageConfig(
                width=4,
                height=4,
                num_agents=2,
                start_positions=[(0, 0), (3, 3)],
                observation_radius=1,
                communication_radius=4,
                use_explicit_map_memory=True,
                share_map_memory=True,
            )
        )
        env.reset()

        logits = compute_cuap_logits(env, CUAPConfig(enabled=True))

        self.assertEqual(logits.shape, (2, 4))
        self.assertTrue(np.isfinite(logits).all())

    def test_cuap_no_agent_id_dependency_by_permutation(self) -> None:
        env = GridCoverageEnv(
            GridCoverageConfig(
                width=5,
                height=1,
                num_agents=2,
                start_positions=[(0, 1), (0, 3)],
                communication_radius=0,
                use_explicit_map_memory=True,
            )
        )
        env.reset()
        env.positions = [(0, 1), (0, 3)]
        env.known_free_by_agent = [{(0, 0), (0, 1), (0, 2)}, {(0, 2), (0, 3), (0, 4)}]
        env.known_team_covered_by_agent = [{(0, 1)}, {(0, 3)}]
        env.known_obstacles_by_agent = [set(), set()]
        env._sync_legacy_aliases()
        original = compute_cuap_logits(env, CUAPConfig(enabled=True, normalize=False))

        env.positions = [(0, 3), (0, 1)]
        env.known_free_by_agent = [{(0, 2), (0, 3), (0, 4)}, {(0, 0), (0, 1), (0, 2)}]
        env.known_team_covered_by_agent = [{(0, 3)}, {(0, 1)}]
        env.known_obstacles_by_agent = [set(), set()]
        env._sync_legacy_aliases()
        permuted = compute_cuap_logits(env, CUAPConfig(enabled=True, normalize=False))

        np.testing.assert_allclose(original[0], permuted[1])
        np.testing.assert_allclose(original[1], permuted[0])

    def test_cuap_respects_action_mask_order(self) -> None:
        env = GridCoverageEnv(
            GridCoverageConfig(
                width=3,
                height=3,
                start=(0, 0),
                obstacles=[(0, 1)],
                observation_radius=1,
                use_explicit_map_memory=True,
            )
        )
        env.reset()
        prior = torch.as_tensor(compute_cuap_logits(env, CUAPConfig(enabled=True))[0], dtype=torch.float32)
        logits = torch.tensor([[10.0, 0.0, 9.0, 8.0]]) + prior.unsqueeze(0)
        masked = ActorCritic._mask_action_logits(logits, torch.as_tensor(env.action_masks(), dtype=torch.bool))

        self.assertEqual(int(torch.argmax(masked, dim=-1).item()), 1)
        self.assertLess(float(masked[0, 0]), -1e30)
        self.assertLess(float(masked[0, 2]), -1e30)
        self.assertLess(float(masked[0, 3]), -1e30)

    def test_ppo_prior_consistency(self) -> None:
        env = GridCoverageEnv(GridCoverageConfig(width=2, height=2, start=(0, 0)))
        observation = env.reset()
        state = env.global_state()
        model = ActorCritic(
            observation_dim=env.observation_dim,
            action_dim=env.action_dim,
            hidden_dim=16,
            state_shape=(env.config.height, env.config.width),
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        prior = torch.tensor([[0.0, 0.0, 0.0, 3.0]], dtype=torch.float32)
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32)
        state_tensor = torch.as_tensor(state, dtype=torch.float32)

        actions, rollout_log_prob, _ = model.act_batch(
            obs_tensor,
            state_tensor,
            action_prior_logits=prior,
            deterministic=True,
        )
        update_log_prob, _, _ = model.evaluate_actions(
            obs_tensor,
            state_tensor,
            actions,
            action_prior_logits=prior,
        )
        plain_log_prob, _, _ = model.evaluate_actions(obs_tensor, state_tensor, actions)

        self.assertTrue(torch.allclose(rollout_log_prob, update_log_prob))
        self.assertFalse(torch.allclose(rollout_log_prob, plain_log_prob))

    def test_gated_cuap_low_margin_prior_not_amplified(self) -> None:
        raw_scores = np.asarray([[0.51, 0.50, 0.49, 0.50]], dtype=np.float32)
        action_masks = np.ones_like(raw_scores, dtype=bool)

        prior = build_bounded_prior(raw_scores, action_masks, tau=1.0)

        self.assertLess(float(np.max(np.abs(prior))), 0.02)

    def test_gated_cuap_single_valid_action_zero_prior(self) -> None:
        raw_scores = np.asarray([[0.0, 1.0, -1.0, 2.0]], dtype=np.float32)
        action_masks = np.asarray([[False, False, False, True]], dtype=bool)

        prior = build_bounded_prior(raw_scores, action_masks, tau=1.0)

        self.assertTrue(np.all(prior == 0.0))

    def test_gated_cuap_return_phase_disables_gate(self) -> None:
        env = GridCoverageEnv(
            GridCoverageConfig(
                width=3,
                height=3,
                use_depot=True,
                depot=(0, 0),
                initial_return_mode=True,
                return_start_positions=[(2, 2)],
                observation_radius=1,
                use_explicit_map_memory=True,
            )
        )
        env.reset()

        inputs = build_cuap_step_inputs(env, CUAPConfig(enabled=True, gated=True), phase="return")

        self.assertTrue(np.all(inputs.phase_mask == 0.0))
        self.assertTrue(np.all(inputs.prior == 0.0))
        self.assertTrue(np.all(inputs.confidence == 0.0))

    def test_gated_cuap_logprob_consistency_and_gate_gradient(self) -> None:
        env = GridCoverageEnv(GridCoverageConfig(width=2, height=2, start=(0, 0)))
        observation = env.reset()
        state = env.global_state()
        model = ActorCritic(
            observation_dim=env.observation_dim,
            action_dim=env.action_dim,
            hidden_dim=16,
            state_shape=(env.config.height, env.config.width),
            use_gated_cuap=True,
            cuap_beta=0.5,
            cuap_gate_hidden_dim=8,
            cuap_gate_init_prob=0.5,
        )
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if not name.startswith("cuap_gate."):
                    parameter.zero_()
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32)
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        cuap_prior = torch.tensor([[0.0, 0.0, 0.0, 4.0]], dtype=torch.float32)
        cuap_confidence = torch.full((1, 2), 0.25, dtype=torch.float32)
        cuap_phase_mask = torch.ones((1, 1), dtype=torch.float32)

        actions, rollout_log_prob, _ = model.act_batch(
            obs_tensor,
            state_tensor,
            cuap_prior=cuap_prior,
            cuap_confidence=cuap_confidence,
            cuap_phase_mask=cuap_phase_mask,
            deterministic=True,
        )
        update_log_prob, _, _ = model.evaluate_actions(
            obs_tensor,
            state_tensor,
            actions,
            cuap_prior=cuap_prior,
            cuap_confidence=cuap_confidence,
            cuap_phase_mask=cuap_phase_mask,
        )
        loss = -update_log_prob.mean()
        model.zero_grad()
        loss.backward()
        gate_grad = sum(
            float(parameter.grad.detach().abs().sum())
            for name, parameter in model.named_parameters()
            if name.startswith("cuap_gate.") and parameter.grad is not None
        )

        self.assertTrue(torch.allclose(rollout_log_prob, update_log_prob))
        self.assertGreater(gate_grad, 0.0)

    def test_gated_cuap_loads_gat_on_checkpoint_missing_gate_only(self) -> None:
        env = GridCoverageEnv(GridCoverageConfig(width=3, height=3, start=(0, 0)))
        plain_model = ActorCritic(
            observation_dim=env.observation_dim,
            action_dim=env.action_dim,
            hidden_dim=16,
            state_shape=(env.config.height, env.config.width),
            use_graph_attention=True,
            gat_num_heads=4,
        )
        gated_model = ActorCritic(
            observation_dim=env.observation_dim,
            action_dim=env.action_dim,
            hidden_dim=16,
            state_shape=(env.config.height, env.config.width),
            use_graph_attention=True,
            gat_num_heads=4,
            use_gated_cuap=True,
            cuap_gate_hidden_dim=8,
            cuap_gate_init_prob=0.1,
        )

        gated_model.load_compatible_state_dict(plain_model.state_dict())

        self.assertIsNotNone(gated_model.cuap_gate)

    def test_cuap_disabled_in_return_phase(self) -> None:
        env = GridCoverageEnv(
            GridCoverageConfig(
                width=3,
                height=3,
                use_depot=True,
                depot=(0, 0),
                initial_return_mode=True,
                return_start_positions=[(2, 2)],
                observation_radius=1,
                use_explicit_map_memory=True,
            )
        )
        env.reset()

        logits = compute_cuap_logits(env, CUAPConfig(enabled=True), phase="return")

        self.assertTrue(np.all(logits == 0.0))

    def test_train_cuap_smoke(self) -> None:
        config = load_config(ROOT / "configs" / "cuap_smoke.toml")
        run_dir = ROOT / ".tmp_tests" / "cuap-smoke"
        shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            checkpoint = train_ppo(config, run_dir=run_dir)

            self.assertTrue(checkpoint.exists())
            self.assertTrue((run_dir / "metrics.csv").exists())
            self.assertTrue((run_dir / "course_config.json").exists())
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_train_gated_cuap_smoke(self) -> None:
        config = load_config(ROOT / "configs" / "gated_cuap_smoke.toml")
        run_dir = ROOT / ".tmp_tests" / "gated-cuap-smoke"
        shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            checkpoint = train_ppo(config, run_dir=run_dir)

            self.assertTrue(checkpoint.exists())
            self.assertTrue((run_dir / "metrics.csv").exists())
            self.assertTrue((run_dir / "course_config.json").exists())
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
