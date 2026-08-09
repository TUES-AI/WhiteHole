"""Render source and shifted Reacher observation domains.

The generated frames reuse the same dataset qpos/qvel states for every domain.
That keeps dynamics/state fixed and isolates the observation shift:

- source: default LeWM Reacher rendering.
- medium_visual: material, lighting, and deterministic color-grade shift.
- hard_camera: oblique fixed MuJoCo camera perspective.
- dcs_bear_static/dynamic: DAVIS bear frames in the MuJoCo sky texture.
- dynamic_camera: per-episode phased camera orbit with changing elevation and zoom.
"""

from __future__ import annotations

import argparse
import os
import sys
from functools import lru_cache
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


VARIANTS = ("source", "medium_visual", "hard_camera", "dynamic_camera", *DCS_VARIANTS)
HARD_CAMERA_POS = np.array([0.16, -0.16, 0.82], dtype=np.float64)
HARD_CAMERA_TARGET = np.array([0.0, 0.0, 0.0], dtype=np.float64)
HARD_CAMERA_FOVY = 42.0
DYNAMIC_CAMERA_PERIOD = 24.0
SOURCE_CAMERA_POS = np.array([0.0, 0.0, 0.75], dtype=np.float64)
SOURCE_CAMERA_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
SOURCE_CAMERA_FOVY = 45.0
CANONICAL_PLANE_Z = 0.015
REACHER_FOREGROUND_GEOMS = ("root", "target", "arm", "hand", "finger")


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


def camera_phase_from_seed(seed: int | np.integer | None) -> float:
    """Map an episode seed to a deterministic camera phase in [0, 2 pi)."""
    if seed is None:
        return 0.0
    seed_value = int(np.asarray(seed).reshape(-1)[0])
    return float(2.0 * np.pi * ((seed_value * 0.7548776662466927) % 1.0))


def dynamic_camera_parameters(
    step: float,
    phase_offset: float = 0.0,
) -> dict[str, np.ndarray | float]:
    """Return a smooth, bounded camera pose for one environment step."""
    phase = phase_offset + 2.0 * np.pi * float(step) / DYNAMIC_CAMERA_PERIOD
    radius = 0.24 + 0.04 * np.sin(2.0 * phase + 0.3)
    height = 0.74 + 0.10 * np.sin(3.0 * phase + 0.6)
    position = np.array(
        [radius * np.cos(phase), radius * np.sin(phase), height],
        dtype=np.float64,
    )
    target = np.array(
        [
            0.018 * np.sin(2.0 * phase),
            0.018 * np.cos(3.0 * phase),
            0.0,
        ],
        dtype=np.float64,
    )
    fovy = 46.0 + 6.0 * np.sin(2.0 * phase + 0.2)
    return {
        "position": position,
        "target": target,
        "fovy": float(fovy),
        "phase": float(phase),
    }


def set_dynamic_camera(
    env: ReacherDMControlWrapper,
    step: float,
    phase_offset: float = 0.0,
) -> None:
    camera_id = 0
    physics = env.env.physics
    params = dynamic_camera_parameters(step, phase_offset)
    position = np.asarray(params["position"])
    target = np.asarray(params["target"])
    physics.model.cam_pos[camera_id] = position
    physics.model.cam_quat[camera_id] = look_at_quat(position, target)
    physics.model.cam_fovy[camera_id] = float(params["fovy"])
    env.camera_id = camera_id


def project_world_to_image(
    points: np.ndarray,
    camera_position: np.ndarray,
    camera_quaternion: np.ndarray,
    fovy: float,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Project world points using MuJoCo's -Z-looking camera convention."""
    rotation = transformations.quat_to_mat(camera_quaternion)[:3, :3]
    camera_points = (np.asarray(points) - camera_position) @ rotation
    depth = -camera_points[:, 2]
    if np.any(depth <= 0.0):
        raise ValueError("Cannot project points behind the camera")

    focal_length = 0.5 * image_height / np.tan(np.deg2rad(fovy) / 2.0)
    return np.column_stack(
        [
            image_width / 2.0 + focal_length * camera_points[:, 0] / depth,
            image_height / 2.0 - focal_length * camera_points[:, 1] / depth,
        ]
    )


def dynamic_to_source_homography(
    step: float,
    phase_offset: float,
    image_width: int,
    image_height: int,
    plane_z: float = CANONICAL_PLANE_Z,
) -> np.ndarray:
    """Map a dynamic-camera workspace plane into the source-camera view."""
    plane_points = np.array(
        [
            [-0.35, -0.35, plane_z],
            [0.35, -0.35, plane_z],
            [0.35, 0.35, plane_z],
            [-0.35, 0.35, plane_z],
        ],
        dtype=np.float64,
    )
    params = dynamic_camera_parameters(step, phase_offset)
    position = np.asarray(params["position"])
    dynamic_pixels = project_world_to_image(
        plane_points,
        position,
        look_at_quat(position, np.asarray(params["target"])),
        float(params["fovy"]),
        image_width,
        image_height,
    )
    source_pixels = project_world_to_image(
        plane_points,
        SOURCE_CAMERA_POS,
        SOURCE_CAMERA_QUAT,
        SOURCE_CAMERA_FOVY,
        image_width,
        image_height,
    )
    return cv2.getPerspectiveTransform(
        dynamic_pixels.astype(np.float32), source_pixels.astype(np.float32)
    )


def canonicalize_dynamic_frame(
    frame: np.ndarray,
    step: float,
    phase_offset: float,
    source_background: np.ndarray | None = None,
) -> np.ndarray:
    """Rectify a dynamic-camera frame, optionally restoring unseen background."""
    image_height, image_width = frame.shape[:2]
    homography = dynamic_to_source_homography(
        step, phase_offset, image_width, image_height
    )
    if source_background is None:
        return cv2.warpPerspective(
            frame,
            homography,
            (image_width, image_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    if source_background.shape != frame.shape:
        raise ValueError(
            "source_background and frame must have the same shape; "
            f"got {source_background.shape} and {frame.shape}"
        )
    warped = cv2.warpPerspective(
        frame,
        homography,
        (image_width, image_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.float32)
    coverage = cv2.warpPerspective(
        np.ones((image_height, image_width), dtype=np.float32),
        homography,
        (image_width, image_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    coverage = np.clip(coverage, 0.0, 1.0)
    completed = warped + source_background.astype(np.float32) * (
        1.0 - coverage[..., None]
    )
    return np.clip(completed, 0.0, 255.0).round().astype(frame.dtype)


@lru_cache(maxsize=4)
def _cached_source_background(image_size: int) -> np.ndarray:
    env = make_env("source")
    physics = env.env.physics
    try:
        hide_reacher_foreground(physics)
        physics.forward()
        background = env.render(width=image_size, height=image_size)
    finally:
        env.close()
    background.setflags(write=False)
    return background


def render_source_background(image_size: int) -> np.ndarray:
    """Render a source-view workspace with all state-dependent geoms hidden."""
    return _cached_source_background(image_size).copy()


def hide_reacher_foreground(physics) -> None:
    """Hide all movable Reacher geoms while retaining the static workspace."""
    for geom_name in REACHER_FOREGROUND_GEOMS:
        geom_id = physics.model.name2id(geom_name, "geom")
        physics.model.geom_rgba[geom_id, 3] = 0.0


def make_env(
    variant: str,
    background_video: str | Path | None = None,
    background_seed: int = 0,
) -> ReacherDMControlWrapper:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {VARIANTS}")
    env = ReacherDMControlWrapper(task="qpos_match", seed=0, render_mode="rgb_array")
    if variant == "medium_visual":
        env.reset(seed=0, options=medium_variation_options())
    else:
        env.reset(seed=0)
    if variant == "hard_camera":
        set_hard_camera(env)
    elif variant == "dynamic_camera":
        set_dynamic_camera(env, step=0.0)
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
    camera_steps: np.ndarray | list[float] | float | None = None,
    camera_phase_offsets: np.ndarray | list[float] | float | None = None,
    render_env: ReacherDMControlWrapper | None = None,
) -> list[np.ndarray]:
    owns_env = render_env is None
    env = (
        render_env
        if render_env is not None
        else make_env(variant, background_video, background_seed)
    )
    num_states = len(states["qpos"])
    if background_episode_ids is None:
        background_episode_ids = np.zeros(num_states, dtype=int)
    if background_step_indices is None:
        background_step_indices = np.arange(num_states)
    if camera_steps is None:
        camera_steps_array = np.arange(num_states, dtype=np.float64)
    else:
        camera_steps_array = np.broadcast_to(
            np.asarray(camera_steps, dtype=np.float64), (num_states,)
        )

    if camera_phase_offsets is None:
        seeds = states.get("seed")
        if seeds is None:
            phase_offsets = np.zeros(num_states, dtype=np.float64)
        else:
            phase_offsets = np.asarray(
                [camera_phase_from_seed(seed) for seed in seeds],
                dtype=np.float64,
            )
    else:
        phase_offsets = np.broadcast_to(
            np.asarray(camera_phase_offsets, dtype=np.float64), (num_states,)
        )

    frames = []
    for index, (qpos, qvel) in enumerate(zip(states["qpos"], states["qvel"])):
        env.set_state(qpos, qvel)
        if is_dcs_variant(variant):
            env._dcs_background.configure_episode(
                int(background_episode_ids[index]),
                int(background_step_indices[index]),
            )
            env._dcs_background.apply(env.env.physics)
        if variant == "dynamic_camera":
            set_dynamic_camera(
                env,
                step=float(camera_steps_array[index]),
                phase_offset=float(phase_offsets[index]),
            )
        frame = env.render(width=image_size, height=image_size)
        if variant == "medium_visual":
            frame = color_grade(frame)
        frames.append(frame)
    if owns_env:
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
