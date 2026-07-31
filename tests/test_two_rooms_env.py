import unittest

import torch

from pldm_envs.wall.wall import DotWall
from pldm_envs.wall.data.wall import WallDataset, WallDatasetConfig


class DotWallTest(unittest.TestCase):
    def make_env(self):
        return DotWall(device=torch.device("cpu"))

    def test_reset_returns_paper_observation_shape(self):
        env = self.make_env()
        obs, info = env.reset(seed=123)

        self.assertEqual(tuple(obs.shape), (2, 64, 64))
        self.assertEqual(obs.dtype, torch.uint8)
        self.assertIn("dot_position", info)
        self.assertIn("target_position", info)
        self.assertIn("target_obs", info)

    def test_reset_seed_is_deterministic(self):
        env = self.make_env()
        _, first = env.reset(seed=7)
        _, second = env.reset(seed=7)

        self.assertTrue(torch.equal(first["dot_position"], second["dot_position"]))
        self.assertTrue(torch.equal(first["target_position"], second["target_position"]))

    def test_action_is_clipped_by_norm(self):
        env = self.make_env()
        env.reset(
            options={
                "location": [10.0, 20.0],
                "target_location": [50.0, 20.0],
            }
        )
        _, reward, done, truncated, info = env.step([10.0, 0.0])

        self.assertAlmostEqual(info["dot_position"][0].item(), 12.45, places=4)
        self.assertAlmostEqual(info["dot_position"][1].item(), 20.0, places=4)
        self.assertEqual(reward, 0.0)
        self.assertIs(done, False)
        self.assertIs(truncated, False)

    def test_wall_blocks_crossing_outside_door(self):
        env = self.make_env()
        env.reset(
            seed=5,
            options={
                "location": [30.0, 30.0],
                "target_location": [50.0, 30.0],
            },
        )
        _, _, _, _, info = env.step([10.0, 0.0])

        self.assertLess(info["dot_position"][0].item(), 31.0)

    def test_agent_can_cross_through_door(self):
        env = self.make_env()
        env.reset(
            seed=5,
            options={
                "location": [30.0, 10.0],
                "target_location": [50.0, 10.0],
            },
        )
        _, _, _, _, info = env.step([10.0, 0.0])

        self.assertGreater(info["dot_position"][0].item(), 32.0)

    def test_reward_is_one_on_goal_reach(self):
        env = self.make_env()
        env.reset(
            options={
                "location": [10.0, 20.0],
                "target_location": [11.0, 20.0],
            }
        )
        _, reward, done, _, _ = env.step([1.0, 0.0])

        self.assertEqual(reward, 1.0)
        self.assertIs(done, True)

    def test_dataset_generator_uses_paper_observation_shape_by_default(self):
        dataset = WallDataset(
            WallDatasetConfig(batch_size=2, n_steps=4, size=2, device="cpu")
        )
        sample = dataset[0]

        self.assertEqual(tuple(sample.states.shape), (2, 4, 2, 64, 64))
        self.assertEqual(sample.states.dtype, torch.uint8)
        self.assertEqual(tuple(sample.actions.shape), (2, 3, 2))
        self.assertEqual(tuple(sample.locations.shape), (2, 4, 2))


if __name__ == "__main__":
    unittest.main()
