import argparse
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whitehole_envs.wall.appearance import APPEARANCE_SHIFTS, apply_appearance_shift
from whitehole_envs.wall.wall import DotWall


def compose_frame(obs, target_obs, target_overlay_obs=None):
    obs = obs.detach().cpu().numpy()
    target_obs = target_obs.detach().cpu().numpy()
    if target_overlay_obs is not None:
        target_overlay_obs = target_overlay_obs.detach().cpu().numpy()

    agent = obs[0].astype(np.float32) / 255.0
    walls = obs[1].astype(np.float32) / 255.0
    target_source = target_overlay_obs if target_overlay_obs is not None else target_obs
    target = target_source[0].astype(np.float32) / 255.0

    frame = np.ones((*agent.shape, 3), dtype=np.float32) * 245.0
    frame[walls > 0.08] = np.array([35.0, 35.0, 35.0])
    frame[..., 1] = np.maximum(frame[..., 1], target * 220.0)
    frame[..., 0] = np.maximum(frame[..., 0], agent * 255.0)
    frame[..., 1] *= 1.0 - agent * 0.5
    frame[..., 2] *= 1.0 - agent * 0.5

    return frame.clip(0, 255).astype(np.uint8)


def make_heuristic_waypoints(env):
    position = env.dot_position.detach().cpu().numpy()
    target = env.target_position.detach().cpu().numpy()
    wall_x = env.wall_x.item()
    door_y = env.hole_y.item()

    current_side = np.sign(position[0] - wall_x)
    target_side = np.sign(target[0] - wall_x)
    crosses_wall = current_side != target_side

    if crosses_wall:
        wall_margin = env.wall_width // 2 + 2.0
        entry = np.array([wall_x + current_side * wall_margin, door_y])
        exit_ = np.array([wall_x - current_side * wall_margin, door_y])
        return [entry, exit_, target]

    return [target]


def action_toward(env, waypoint):
    position = env.dot_position.detach().cpu().numpy()
    delta = waypoint - position
    norm = np.linalg.norm(delta)
    if norm < 1e-6:
        return np.zeros(2, dtype=np.float32)

    return (delta / norm * min(norm, env.max_step_norm)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Visualize the WhiteHole Two-Rooms env.")
    parser.add_argument("--output-dir", default="outputs/two_rooms")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument(
        "--appearance-shift",
        choices=APPEARANCE_SHIFTS,
        default="source",
        help="Render the source observation or an appearance-shifted target view.",
    )
    parser.add_argument(
        "--policy",
        choices=["heuristic", "random"],
        default="heuristic",
        help="Use a door-seeking heuristic or random actions for the rollout.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = DotWall(device=torch.device("cpu"))
    obs, info = env.reset(seed=args.seed)
    target_obs = info["target_obs"]

    shifted_obs = apply_appearance_shift(obs, args.appearance_shift)
    shifted_target_obs = apply_appearance_shift(target_obs, args.appearance_shift)

    frames = [compose_frame(shifted_obs, shifted_target_obs, target_obs)]
    imageio.imwrite(output_dir / "initial.png", frames[0])

    done = truncated = False
    waypoints = make_heuristic_waypoints(env)
    for _ in range(args.steps):
        if args.policy == "random":
            action = env.action_space.sample()
        else:
            while len(waypoints) > 1:
                position = env.dot_position.detach().cpu().numpy()
                if np.linalg.norm(position - waypoints[0]) > 0.75:
                    break
                waypoints.pop(0)
            action = action_toward(env, waypoints[0])

        obs, _, done, truncated, info = env.step(action)
        shifted_obs = apply_appearance_shift(obs, args.appearance_shift)
        shifted_target_obs = apply_appearance_shift(
            info["target_obs"], args.appearance_shift
        )
        frames.append(
            compose_frame(shifted_obs, shifted_target_obs, info["target_obs"])
        )
        if done or truncated:
            break

    imageio.mimsave(output_dir / "rollout.gif", frames, fps=args.fps)

    print(f"Appearance shift: {args.appearance_shift}")
    print(f"Wrote {output_dir / 'initial.png'}")
    print(f"Wrote {output_dir / 'rollout.gif'}")
    print(f"Final position: {info['dot_position'].detach().cpu().numpy().round(2)}")
    print(f"Target position: {info['target_position'].detach().cpu().numpy().round(2)}")
    print(f"Done: {done}; truncated: {truncated}; frames: {len(frames)}")


if __name__ == "__main__":
    main()
