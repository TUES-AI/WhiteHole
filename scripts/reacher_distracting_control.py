"""Distracting Control Suite-style DAVIS backgrounds for Reacher.

This reproduces the paper-era DCS background mechanism without its TensorFlow
input pipeline: DAVIS RGB frames replace MuJoCo's sky texture and Reacher's
floor is made transparent. Dynamics-relevant simulator data are never modified.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from dm_control.mujoco.wrapper import mjbindings

DCS_VARIANTS = ("dcs_bear_static", "dcs_bear_dynamic")
DAVIS_HF_REPO = "emirkisa/DAVIS-2017-480p-mp4"
DAVIS_HF_FILENAME = "bear_raw_24fps.mp4"
DAVIS_HF_REVISION = "872ccd3c5d4d9c98016798e7af0342b33ec6ab4d"
SKY_TEXTURE_INDEX = 0
DCS_SKY_HEIGHT = 800
DCS_REACHER_GROUND_ALPHA = 0.0


def is_dcs_variant(variant: str) -> bool:
    return variant in DCS_VARIANTS


def resolve_bear_video(path: str | Path | None = None) -> Path:
    """Resolve an explicit path, DCS_BACKGROUND_VIDEO, or cached HF mirror."""
    candidate = path or os.environ.get("DCS_BACKGROUND_VIDEO")
    if candidate:
        resolved = Path(candidate).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"DAVIS background video not found: {resolved}")
        return resolved

    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=DAVIS_HF_REPO,
            filename=DAVIS_HF_FILENAME,
            repo_type="dataset",
            revision=DAVIS_HF_REVISION,
        )
    ).absolute()


@lru_cache(maxsize=4)
def _load_video(path_string: str) -> tuple[np.ndarray, float]:
    capture = cv2.VideoCapture(path_string)
    if not capture.isOpened():
        raise ValueError(f"Could not decode DAVIS video: {path_string}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise ValueError(f"DAVIS video contains no decodable frames: {path_string}")
    return np.stack(frames), fps


@lru_cache(maxsize=256)
def _resized_frame(
    path_string: str, frame_index: int, height: int, width: int
) -> np.ndarray:
    frames, _ = _load_video(path_string)
    frame = frames[frame_index % len(frames)]
    # The official DCS wrapper resizes each DAVIS frame directly to the square
    # MuJoCo sky texture rather than preserving its aspect ratio.
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def _episode_parameters(
    seed: int, episode_id: int, frame_count: int
) -> tuple[int, int]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, episode_id]))
    return int(rng.integers(frame_count)), int(rng.choice((-1, 1)))


def bouncing_frame_index(
    frame_count: int, offset: int, direction: int, step: int
) -> int:
    """DCS-style bidirectional playback with reflection at video endpoints."""
    if frame_count <= 1:
        return 0
    # The official wrapper clamps after stepping out of range, so each endpoint
    # is displayed twice before direction reverses.
    period = 2 * frame_count
    phase = offset if direction > 0 else period - 1 - offset
    position = (phase + step) % period
    return int(min(position, period - 1 - position))


class DavisBearBackground:
    """Apply one deterministic static or dynamic bear video to MuJoCo's sky."""

    def __init__(
        self,
        video_path: str | Path | None,
        *,
        dynamic: bool,
        seed: int = 0,
    ) -> None:
        self.video_path = resolve_bear_video(video_path)
        self.dynamic = dynamic
        self.seed = int(seed)
        self.episode_id = 0
        self.episode_step = 0
        self._offset = 0
        self._direction = 1
        self._last_applied_index: int | None = None
        self._initialized_physics: set[int] = set()
        self.frames, self.fps = _load_video(str(self.video_path))
        self.configure_episode(0, 0)

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def frame_index(self) -> int:
        step = self.episode_step if self.dynamic else 0
        return bouncing_frame_index(
            self.frame_count, self._offset, self._direction, step
        )

    def configure_episode(self, episode_id: int, step: int = 0) -> None:
        self.episode_id = int(episode_id)
        self.episode_step = int(step)
        self._offset, self._direction = _episode_parameters(
            self.seed, self.episode_id, self.frame_count
        )
        self._last_applied_index = None

    def advance(self, physics, steps: int = 1) -> None:
        if self.dynamic:
            self.episode_step += int(steps)
        self.apply(physics)

    def _initialize_physics(self, physics) -> None:
        physics_id = id(physics)
        if physics_id in self._initialized_physics:
            return
        physics.named.model.mat_rgba["grid", "a"] = DCS_REACHER_GROUND_ALPHA
        physics.model.tex_height[SKY_TEXTURE_INDEX] = DCS_SKY_HEIGHT
        self._initialized_physics.add(physics_id)

    def apply(self, physics, force: bool = False) -> None:
        self._initialize_physics(physics)
        index = self.frame_index
        if not force and index == self._last_applied_index:
            return

        model = physics.model
        height = int(model.tex_height[SKY_TEXTURE_INDEX])
        width = int(model.tex_width[SKY_TEXTURE_INDEX])
        channels = int(model.tex_nchannel[SKY_TEXTURE_INDEX])
        if channels != 3:
            raise ValueError(f"Expected a 3-channel sky texture, found {channels}")
        address = int(model.tex_adr[SKY_TEXTURE_INDEX])
        size = height * width * channels
        texture_store = getattr(model, "tex_data", None)
        if texture_store is None:
            texture_store = model.tex_rgb
        texture = _resized_frame(str(self.video_path), index, height, width)
        texture_store[address : address + size] = texture.reshape(-1)

        with physics.contexts.gl.make_current() as context:
            context.call(
                mjbindings.mjlib.mjr_uploadTexture,
                model.ptr,
                physics.contexts.mujoco.ptr,
                SKY_TEXTURE_INDEX,
            )
        self._last_applied_index = index

    def metadata(self) -> dict:
        stat = self.video_path.stat()
        return {
            "video_path": str(self.video_path),
            "video_filename": self.video_path.name,
            "sha256": hashlib.sha256(self.video_path.read_bytes()).hexdigest(),
            "bytes": stat.st_size,
            "frame_count": self.frame_count,
            "resolution": [int(self.frames.shape[2]), int(self.frames.shape[1])],
            "fps": self.fps,
            "dynamic": self.dynamic,
            "seed": self.seed,
            "ground_plane_alpha": DCS_REACHER_GROUND_ALPHA,
            "sky_texture_height": DCS_SKY_HEIGHT,
            "source": (
                f"https://huggingface.co/datasets/{DAVIS_HF_REPO}/resolve/"
                f"{DAVIS_HF_REVISION}/{DAVIS_HF_FILENAME}"
            ),
        }
