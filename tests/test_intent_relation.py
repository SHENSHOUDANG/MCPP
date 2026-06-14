from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mathbased_mcpp.env import GridCoverageEnv
from mathbased_mcpp.config import GridCoverageConfig
from mathbased_mcpp.intent_relation import intent_overlap_from_node_messages, pairwise_soft_iou
from mathbased_mcpp.ppo import ActorCritic, GraphAttentionBlock


class IntentRelationTests(unittest.TestCase):
    def test_pairwise_soft_iou_handles_identity_empty_and_disjoint(self) -> None:
        regions = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ]
        )

        overlap = pairwise_soft_iou(regions)

        self.assertAlmostEqual(float(overlap[0, 0, 1]), 1.0)
        self.assertAlmostEqual(float(overlap[0, 0, 2]), 0.0)
        self.assertAlmostEqual(float(overlap[0, 3, 3]), 0.0)
        self.assertTrue(torch.isfinite(overlap).all())

    def test_node_message_overlap_respects_self_and_communication_mask(self) -> None:
        messages = torch.zeros(3, 24)
        region_start = 14
        messages[0, region_start] = 1.0
        messages[1, region_start] = 1.0
        messages[2, region_start + 1] = 1.0
        messages[:, -1] = 1.0
        comm_mask = torch.tensor(
            [
                [True, True, False],
                [True, True, True],
                [False, True, True],
            ]
        )

        overlap = intent_overlap_from_node_messages(messages, comm_mask, intent_grid_size=3)

        self.assertAlmostEqual(float(overlap[0, 0]), 0.0)
        self.assertAlmostEqual(float(overlap[0, 1]), 1.0)
        self.assertAlmostEqual(float(overlap[0, 2]), 0.0)
        self.assertAlmostEqual(float(overlap[1, 2]), 0.0)

    def test_beta_zero_cir_matches_plain_gat_mappo_logits(self) -> None:
        env = GridCoverageEnv(
            GridCoverageConfig(
                width=5,
                height=5,
                num_agents=3,
                start_positions=[(0, 0), (0, 4), (4, 0)],
                observation_radius=1,
                communication_radius=4,
                use_explicit_map_memory=True,
                share_map_memory=True,
                intent_grid_size=3,
            )
        )
        observation = torch.as_tensor(env.reset(), dtype=torch.float32)
        state = torch.as_tensor(env.global_state(), dtype=torch.float32)
        neighbor_mask = torch.as_tensor(env.neighbor_mask(), dtype=torch.bool)
        edge_features = torch.as_tensor(env.neighbor_features(), dtype=torch.float32)
        node_messages = torch.as_tensor(env.node_messages(), dtype=torch.float32)
        kwargs = dict(
            observation_dim=env.observation_dim,
            action_dim=env.action_dim,
            hidden_dim=24,
            state_shape=(env.config.height, env.config.width),
            state_channels=env.state_channels,
            state_metadata_dim=env.state_metadata_dim,
            use_graph_attention=True,
            gat_num_heads=4,
            gat_edge_dim=env.neighbor_feature_dim,
            gat_residual=True,
            node_message_dim=env.node_message_dim,
        )
        plain = ActorCritic(**kwargs)
        cir = ActorCritic(
            **kwargs,
            use_intent_relation=True,
            intent_relation_beta_max=2.0,
            intent_grid_size=env.config.intent_grid_size,
        )
        cir.load_compatible_state_dict(plain.state_dict())

        with torch.no_grad():
            plain_logits = plain._policy_logits(
                observation,
                state,
                neighbor_mask=neighbor_mask,
                edge_features=edge_features,
                node_messages=node_messages,
            )
            cir_logits = cir._policy_logits(
                observation,
                state,
                neighbor_mask=neighbor_mask,
                edge_features=edge_features,
                node_messages=node_messages,
            )
            plain_value = plain.value(state)
            cir_value = cir.value(state)

        self.assertTrue(torch.allclose(plain_logits, cir_logits, atol=1e-6))
        self.assertTrue(torch.allclose(plain_value, cir_value, atol=1e-6))

    def test_intent_relation_bias_changes_attention_direction(self) -> None:
        block = GraphAttentionBlock(
            hidden_dim=4,
            num_heads=1,
            use_intent_relation=True,
            intent_relation_beta_max=2.0,
        )
        features = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            ]
        )
        mask = torch.ones(1, 3, 3, dtype=torch.bool)
        overlap = torch.tensor([[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])

        with torch.no_grad():
            block.query.weight.zero_()
            block.query.bias.zero_()
            block.key.weight.zero_()
            block.key.bias.zero_()
            block.raw_intent_bias.fill_(0.54930615)
        block(features, mask, intent_overlap=overlap)
        positive_attention = block.last_attention_weights[0, 0, 0].clone()

        with torch.no_grad():
            block.raw_intent_bias.fill_(-0.54930615)
        block(features, mask, intent_overlap=overlap)
        negative_attention = block.last_attention_weights[0, 0, 0].clone()

        self.assertGreater(float(positive_attention[1]), float(positive_attention[2]))
        self.assertLess(float(negative_attention[1]), float(positive_attention[1]))


if __name__ == "__main__":
    unittest.main()
