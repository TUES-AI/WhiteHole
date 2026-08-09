"""Verify that the DAVIS bear shift changes pixels, not Reacher state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reacher_distracting_control import DavisBearBackground
from scripts.visualize_reacher_shifts import label_frame, make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--background-video", default=None)
    parser.add_argument("--background-seed", type=int, default=0)
    parser.add_argument("--episode", type=int, default=7)
    parser.add_argument("--step", type=int, default=11)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_env = make_env("source")
    qpos = source_env.env.physics.data.qpos.copy()
    qvel = source_env.env.physics.data.qvel.copy()
    source_env.set_state(qpos, qvel)
    source = source_env.render(width=args.image_size, height=args.image_size)
    source_env.close()

    bear_env = make_env(
        "dcs_bear_dynamic", args.background_video, args.background_seed
    )
    physics = bear_env.env.physics
    bear_env.set_state(qpos, qvel)
    state_before = {
        "qpos": physics.data.qpos.copy(),
        "qvel": physics.data.qvel.copy(),
        "qacc": physics.data.qacc.copy(),
        "ctrl": physics.data.ctrl.copy(),
        "act": physics.data.act.copy(),
        "time": np.asarray(physics.data.time).copy(),
    }
    background: DavisBearBackground = bear_env._dcs_background
    background.configure_episode(args.episode, args.step)
    background.apply(physics)
    bear = bear_env.render(width=args.image_size, height=args.image_size)
    state_after = {
        "qpos": physics.data.qpos.copy(),
        "qvel": physics.data.qvel.copy(),
        "qacc": physics.data.qacc.copy(),
        "ctrl": physics.data.ctrl.copy(),
        "act": physics.data.act.copy(),
        "time": np.asarray(physics.data.time).copy(),
    }

    # Discard the renderer's one-time warm-up frame before the exact check.
    background.configure_episode(args.episode, args.step)
    background.apply(physics, force=True)
    bear = bear_env.render(width=args.image_size, height=args.image_size)
    background.configure_episode(args.episode, args.step)
    background.apply(physics, force=True)
    repeated = bear_env.render(width=args.image_size, height=args.image_size)
    background.configure_episode(args.episode, args.step + 1)
    background.apply(physics, force=True)
    bear_next = bear_env.render(width=args.image_size, height=args.image_size)
    bear_env.close()

    state_unchanged = all(
        np.array_equal(state_before[key], state_after[key]) for key in state_before
    )
    deterministic = np.array_equal(bear, repeated)
    dynamic_changed = not np.array_equal(bear, bear_next)
    if not state_unchanged:
        raise AssertionError("Background rendering changed simulator state")
    if not deterministic:
        raise AssertionError("Repeated background rendering was not deterministic")
    if not dynamic_changed:
        raise AssertionError("Consecutive dynamic background frames were identical")

    contact = np.concatenate(
        [
            label_frame(source, "source"),
            label_frame(bear, f"bear step {args.step}"),
            label_frame(bear_next, f"bear step {args.step + 1}"),
        ],
        axis=1,
    )
    imageio.imwrite(output_dir / "pixel_only_check.png", contact)
    report = {
        "state_unchanged": state_unchanged,
        "state_fields_checked": list(state_before),
        "deterministic_same_state_frame": deterministic,
        "dynamic_next_frame_changed": dynamic_changed,
        "source_to_bear_mean_absolute_pixel_difference": float(
            np.abs(source.astype(np.float32) - bear.astype(np.float32)).mean()
        ),
        "bear_consecutive_mean_absolute_pixel_difference": float(
            np.abs(bear.astype(np.float32) - bear_next.astype(np.float32)).mean()
        ),
        "qpos": qpos.tolist(),
        "qvel": qvel.tolist(),
        "episode": args.episode,
        "step": args.step,
        "background": background.metadata(),
    }
    (output_dir / "verification.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
