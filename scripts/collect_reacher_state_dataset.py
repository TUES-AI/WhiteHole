"""Collect a compact random-policy Reacher state/action dataset.

The public LeWM checkpoint is available without authentication, while the
original ``dmc/reacher_random`` dataset may not be. This collector stores only
the simulator quantities required by the shift/adaptation scripts; observations
are deterministically re-rendered from qpos/qvel during each experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from stable_worldmodel.envs.dmcontrol.reacher import ReacherDMControlWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--episode-length", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3072)
    return parser.parse_args()


def collect(episodes: int, episode_length: int, seed: int) -> dict[str, np.ndarray]:
    env = ReacherDMControlWrapper(task="qpos_match", seed=seed)
    action_rng = np.random.default_rng(seed)
    columns: dict[str, list] = {
        "episode_idx": [],
        "step_idx": [],
        "seed": [],
        "qpos": [],
        "qvel": [],
        "action": [],
    }

    for episode in range(episodes):
        episode_seed = seed + episode
        _, info = env.reset(seed=episode_seed)
        for step in range(episode_length):
            action = action_rng.uniform(-1.0, 1.0, size=env.action_space.shape)
            columns["episode_idx"].append(episode)
            columns["step_idx"].append(step)
            columns["seed"].append(episode_seed)
            columns["qpos"].append(np.asarray(info["qpos"], dtype=np.float64))
            columns["qvel"].append(np.asarray(info["qvel"], dtype=np.float64))
            columns["action"].append(action.astype(np.float64))
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                raise RuntimeError(
                    f"Reacher ended early at episode={episode}, step={step}"
                )
    env.close()
    return {key: np.asarray(values) for key, values in columns.items()}


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.episode_length <= 0:
        raise ValueError("episodes and episode-length must be positive")
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = collect(args.episodes, args.episode_length, args.seed)
    ep_len = np.full(args.episodes, args.episode_length, dtype=np.int64)
    ep_offset = np.arange(args.episodes, dtype=np.int64) * args.episode_length

    with h5py.File(output, "w") as handle:
        handle.create_dataset("ep_len", data=ep_len)
        handle.create_dataset("ep_offset", data=ep_offset)
        for key, values in columns.items():
            handle.create_dataset(key, data=values, compression="gzip")
        handle.attrs["collection"] = json.dumps(
            {
                "environment": "swm/ReacherDMControl-v0",
                "task": "qpos_match",
                "policy": "uniform_random_actions",
                "episodes": args.episodes,
                "episode_length": args.episode_length,
                "seed": args.seed,
                "pixels_stored": False,
            }
        )

    print(f"wrote: {output}")
    print(f"transitions: {len(columns['step_idx'])}")


if __name__ == "__main__":
    main()
