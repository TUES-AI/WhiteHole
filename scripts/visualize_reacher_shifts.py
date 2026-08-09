"""Render source and shifted Reacher observation domains.

The generated frames reuse the same dataset qpos/qvel states for every domain.
That keeps dynamics/state fixed and isolates the observation shift:

- source: default LeWM Reacher rendering.
- medium_visual: material, lighting, and deterministic color-grade shift.
- hard_camera: oblique fixed MuJoCo camera perspective.
- dcs_bear_static/dynamic: DAVIS bear frames in the MuJoCo sky texture.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import stable_worldmodel as swm
from dm_control.utils import transformations
from stable_worldmodel.envs.dmcontrol.reacher import ReacherDMControlWrapper

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reacher_distracting_control import (
    DCS_VARIANTS,
    DavisBearBackground,
    is_dcs_variant,
)


VARIANTS = ("source", "medium_visual", "hard_camera", *DCS_VARIANTS)
HARD_CAMERA_POS = np.array([0.16, -0.16, 0.82], dtype=np.float64)
HARD_CAMERA_TARGET = np.array([0.0, 0.0, 0.0], dtype=np.float64)
HARD_CAMERA_FOVY = 42.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Reacher source and appearance/camera shifts."
    )
    parser.add_argument("--dataset-name", default="dmc/reacher_random")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", default="tmp_reacher_visualization/shifts")
    parser.add_argument("--episode", type=int, default=8506)
    parser.add_argument("--start-step", type=int, default=46)
    parser.add_argument("--num-frames", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--contact-frames", type=int, default=6)
    parser.add_argument(
        "--background-video",
        default=None,
        help=(
            "DAVIS bear MP4. Defaults to DCS_BACKGROUND_VIDEO or downloads "
            "the pinned bear_raw_24fps.mp4 mirror into the Hugging Face cache."
        ),
    )
    parser.add_argument("--background-seed", type=int, default=0)
    return parser.parse_args()


def resolve_cache_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    if os.environ.get("STABLEWM_HOME"):
        return Path(os.environ["STABLEWM_HOME"])
    return Path("stablewm_home")


def get_episode_window(dataset, episode: int, start_step: int, num_frames: int):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_ids = dataset.get_col_data(col_name)
    step_ids = dataset.get_col_data("step_idx")
    stop_step = start_step + num_frames
    mask = (
        (episode_ids == episode)
        & (step_ids >= start_step)
        & (step_ids < stop_step)
    )
    row_indices = np.nonzero(mask)[0]
    if len(row_indices) != num_frames:
        raise ValueError(
            f"Expected {num_frames} rows for episode={episode}, "
            f"start_step={start_step}; found {len(row_indices)}."
        )
    order = np.argsort(step_ids[row_indices])
    return dataset.get_row_data(row_indices[order])


def medium_variation_options() -> dict:
    return {
        "variation_values": {
            "agent.color": np.array([0.22, 0.82, 0.74], dtype=np.float64),
            "floor.color": np.array(
                [[0.19, 0.16, 0.26], [0.34, 0.40, 0.48]],
                dtype=np.float64,
            ),
            "light.intensity": np.array([0.95], dtype=np.float64),
        }
    }


def color_grade(frame: np.ndarray) -> np.ndarray:
    """Apply a fixed, moderate pixel-domain appearance shift."""
    f = frame.astype(np.float32) / 255.0
    mean = f.mean(axis=(0, 1), keepdims=True)
    f = (f - mean) * 1.12 + mean
    f = np.clip(f * 1.10 + np.array([0.015, 0.005, -0.010]), 0.0, 1.0)

    gray = f.mean(axis=2, keepdims=True)
    f = np.clip(gray + (f - gray) * 1.20, 0.0, 1.0)
    return (f * 255.0).round().astype(np.uint8)


def look_at_quat(
    pos: np.ndarray,
    target: np.ndarray = HARD_CAMERA_TARGET,
    up: np.ndarray = np.array([0.0, 1.0, 0.0], dtype=np.float64),
) -> np.ndarray:
    """Return a MuJoCo camera quaternion that looks from pos toward target."""
    forward = target - pos
    forward /= np.linalg.norm(forward)

    z_axis = -forward
    x_axis = np.cross(up, z_axis)
    if np.linalg.norm(x_axis) < 1e-8:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    rotation = np.column_stack([x_axis, y_axis, z_axis])
    return transformations.mat_to_quat(rotation)


def set_hard_camera(env: ReacherDMControlWrapper) -> None:
    camera_id = 0
    physics = env.env.physics
    physics.model.cam_pos[camera_id] = HARD_CAMERA_POS
    physics.model.cam_quat[camera_id] = look_at_quat(HARD_CAMERA_POS)
    physics.model.cam_fovy[camera_id] = HARD_CAMERA_FOVY
    env.camera_id = camera_id


def make_env(
    variant: str,
    background_video: str | Path | None = None,
    background_seed: int = 0,
) -> ReacherDMControlWrapper:
    env = ReacherDMControlWrapper(task="qpos_match", seed=0, render_mode="rgb_array")
    if variant == "medium_visual":
        env.reset(seed=0, options=medium_variation_options())
    else:
        env.reset(seed=0)
    if variant == "hard_camera":
        set_hard_camera(env)
    if is_dcs_variant(variant):
        env._dcs_background = DavisBearBackground(
            background_video,
            dynamic=variant == "dcs_bear_dynamic",
            seed=background_seed,
        )
    return env


def render_variant(
    states: dict,
    variant: str,
    image_size: int,
    background_video: str | Path | None = None,
    background_seed: int = 0,
    background_episode_ids: np.ndarray | list[int] | None = None,
    background_step_indices: np.ndarray | list[int] | None = None,
) -> list[np.ndarray]:
    env = make_env(variant, background_video, background_seed)
    num_states = len(states["qpos"])
    if background_episode_ids is None:
        background_episode_ids = np.zeros(num_states, dtype=int)
    if background_step_indices is None:
        background_step_indices = np.arange(num_states)

    frames = []
    for index, (qpos, qvel) in enumerate(zip(states["qpos"], states["qvel"])):
        env.set_state(qpos, qvel)
        if is_dcs_variant(variant):
            env._dcs_background.configure_episode(
                int(background_episode_ids[index]),
                int(background_step_indices[index]),
            )
            env._dcs_background.apply(env.env.physics)
        frame = env.render(width=image_size, height=image_size)
        if variant == "medium_visual":
            frame = color_grade(frame)
        frames.append(frame)
    env.close()
    return frames


def label_frame(frame: np.ndarray, label: str) -> np.ndarray:
    out = np.pad(frame, ((24, 0), (0, 0), (0, 0)), constant_values=245)
    cv2.putText(
        out,
        label,
        (8, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (45, 45, 45),
        1,
        cv2.LINE_AA,
    )
    return out


def save_contact_sheet(
    frames_by_variant: dict[str, list[np.ndarray]],
    out_path: Path,
    contact_frames: int,
) -> None:
    num_frames = len(next(iter(frames_by_variant.values())))
    sample_indices = np.linspace(
        0, max(0, num_frames - 1), contact_frames, dtype=int
    )
    rows = []
    for idx in sample_indices:
        row = [
            label_frame(frames_by_variant[name][idx], name)
            for name in VARIANTS
        ]
        rows.append(np.concatenate(row, axis=1))
    imageio.imwrite(out_path, np.concatenate(rows, axis=0))


def main() -> None:
    args = parse_args()
    cache_dir = resolve_cache_dir(args.cache_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = swm.data.load_dataset(args.dataset_name, cache_dir=str(cache_dir))
    states = get_episode_window(
        dataset,
        episode=args.episode,
        start_step=args.start_step,
        num_frames=args.num_frames,
    )

    episode_key = (
        "episode_idx" if "episode_idx" in states else "ep_idx"
    )
    render_kwargs = {
        "background_video": args.background_video,
        "background_seed": args.background_seed,
        "background_episode_ids": states[episode_key],
        "background_step_indices": states["step_idx"],
    }
    frames_by_variant = {
        variant: render_variant(states, variant, args.image_size, **render_kwargs)
        for variant in VARIANTS
    }

    for variant, frames in frames_by_variant.items():
        imageio.mimsave(out_dir / f"{variant}.mp4", frames, fps=args.fps)

    save_contact_sheet(
        frames_by_variant,
        out_dir / "reacher_shift_contact_sheet.png",
        args.contact_frames,
    )

    print(f"output_dir: {out_dir.resolve()}")
    print(f"dataset: {args.dataset_name}")
    print(f"cache_dir: {cache_dir.resolve()}")
    print(f"episode: {args.episode}")
    print(f"start_step: {args.start_step}")
    print(f"num_frames: {args.num_frames}")
    print(f"variants: {', '.join(VARIANTS)}")


if __name__ == "__main__":
    main()
