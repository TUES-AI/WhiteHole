"""Evaluate LeWM Reacher under source and shifted visual domains."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reacher_distracting_control import (
    DavisBearBackground,
    is_dcs_variant,
)
from scripts.visualize_reacher_shifts import (
    VARIANTS,
    camera_phase_from_seed,
    canonicalize_dynamic_frame,
    color_grade,
    make_env as make_render_env,
    medium_variation_options,
    render_source_background,
    render_variant,
    set_dynamic_camera,
    set_hard_camera,
)
from scripts.reacher_conv_adapter import AdaptedLeWM, build_input_adapter
from scripts.reacher_movie_adapter import build_movie_encoder


DYNAMIC_CAMERA_VARIANTS = (
    "dynamic_camera",
    "dynamic_camera_homography",
    "dynamic_camera_oracle",
)
EVAL_VARIANTS = VARIANTS + DYNAMIC_CAMERA_VARIANTS[1:]


class ReacherRenderShiftWrapper(gym.Wrapper):
    """Apply a visual-domain shift while leaving Reacher state/dynamics intact."""

    def __init__(
        self,
        env: gym.Env,
        variant: str,
        background_video: str | None = None,
        background_seed: int = 0,
    ):
        super().__init__(env)
        self.variant = variant
        self.background = (
            DavisBearBackground(
                background_video,
                dynamic=variant == "dcs_bear_dynamic",
                seed=background_seed,
            )
            if is_dcs_variant(variant)
            else None
        )
        self.camera_step = 0
        self.camera_phase_offset = 0.0
        self.source_background = (
            render_source_background(224)
            if variant == "dynamic_camera_oracle"
            else None
        )

    def reset(self, *args, **kwargs):
        self.camera_step = 0
        self.camera_phase_offset = camera_phase_from_seed(kwargs.get("seed"))
        if self.variant == "medium_visual":
            options = dict(kwargs.get("options") or {})
            values = dict(options.get("variation_values") or {})
            values.update(medium_variation_options()["variation_values"])
            options["variation_values"] = values
            kwargs["options"] = options

        obs, info = self.env.reset(*args, **kwargs)
        if self.variant == "hard_camera":
            set_hard_camera(self.env.unwrapped)
        elif self.variant in DYNAMIC_CAMERA_VARIANTS:
            set_dynamic_camera(
                self.env.unwrapped,
                step=self.camera_step,
                phase_offset=self.camera_phase_offset,
            )
        if self.background is not None:
            self.background.configure_episode(0, 0)
            self.background.apply(self.env.unwrapped.env.physics)
        return obs, info

    def configure_background(self, episode_id: int, step: int) -> None:
        if self.background is not None:
            self.background.configure_episode(episode_id, step)
            self.background.apply(self.env.unwrapped.env.physics)

    def step(self, action):
        result = self.env.step(action)
        self.camera_step += 1
        if self.background is not None:
            self.background.advance(self.env.unwrapped.env.physics)
        return result

    def render(self, *args, **kwargs):
        if self.variant == "hard_camera":
            set_hard_camera(self.env.unwrapped)
        elif self.variant in DYNAMIC_CAMERA_VARIANTS:
            set_dynamic_camera(
                self.env.unwrapped,
                step=self.camera_step,
                phase_offset=self.camera_phase_offset,
            )
        if self.background is not None:
            self.background.apply(self.env.unwrapped.env.physics)
        frame = self.env.render(*args, **kwargs)
        if self.variant == "medium_visual":
            frame = color_grade(frame)
        elif self.variant in DYNAMIC_CAMERA_VARIANTS[1:]:
            background = self.source_background
            if background is not None and background.shape != frame.shape:
                background = render_source_background(frame.shape[0])
            frame = canonicalize_dynamic_frame(
                frame,
                step=self.camera_step,
                phase_offset=self.camera_phase_offset,
                source_background=background,
            )
        return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LeWM Reacher on source and shifted visual domains."
    )
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--num-eval", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-name", default="dmc/reacher_random")
    parser.add_argument("--policy", default="quentinll/lewm-reacher")
    parser.add_argument("--adapter-checkpoint", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--output-json",
        default="tmp_reacher_visualization/shift_success_rates_30.json",
    )
    parser.add_argument("--goal-offset-steps", type=int, default=25)
    parser.add_argument("--eval-budget", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--receding-horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--cem-steps", type=int, default=30)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--background-video", default=None)
    parser.add_argument("--background-seed", type=int, default=0)
    parser.add_argument("--episode-min", type=int, default=None)
    parser.add_argument("--episode-max", type=int, default=None)
    parser.add_argument("--exclude-episodes", nargs="*", type=int, default=[])
    return parser.parse_args()


def load_adapter_model(model, checkpoint_path: str | None, device: str):
    if checkpoint_path is None:
        return model, None

    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get("adapter_config", {})
    if cfg.get("type") == "encoder":
        model.encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
        model = model.to(device).eval()
        model.requires_grad_(False)
        return model, checkpoint

    if cfg.get("type") == "movie_stn":
        model.encoder = build_movie_encoder(model.encoder, cfg).to(device)
        model.encoder.load_state_dict(checkpoint["sae_state_dict"], strict=True)
        model.projector.load_state_dict(
            checkpoint["projector_state_dict"], strict=True
        )
        model = model.to(device).eval()
        model.requires_grad_(False)
        return model, checkpoint

    if cfg.get("type", "conv") != "conv":
        raise ValueError(f"Unknown adapter checkpoint type: {cfg.get('type')!r}")

    adapter = build_input_adapter(cfg).to(device)
    adapter.load_state_dict(checkpoint["adapter_state_dict"], strict=True)
    adapter.eval()
    adapter.requires_grad_(False)
    return AdaptedLeWM(model, adapter).to(device).eval(), checkpoint


def resolve_cache_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    if os.environ.get("STABLEWM_HOME"):
        return Path(os.environ["STABLEWM_HOME"])
    return Path("stablewm_home")


def img_transform(img_size: int = 224):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    return np.array(
        [np.max(step_idx[episode_idx == ep_id]) + 1 for ep_id in episodes]
    )


def sample_eval_points(
    dataset,
    seed: int,
    num_eval: int,
    goal_offset_steps: int,
    episode_min: int | None = None,
    episode_max: int | None = None,
    exclude_episodes: list[int] | None = None,
):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - goal_offset_steps - 1
    max_start_idx_dict = {
        ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)
    }
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    row_episodes = dataset.get_col_data(col_name)
    if episode_min is not None:
        valid_mask &= row_episodes >= episode_min
    if episode_max is not None:
        valid_mask &= row_episodes <= episode_max
    if exclude_episodes:
        valid_mask &= ~np.isin(row_episodes, exclude_episodes)
    valid_indices = np.nonzero(valid_mask)[0]

    rng = np.random.default_rng(seed)
    if num_eval > len(valid_indices):
        raise ValueError(
            f"Requested num_eval={num_eval}, but only {len(valid_indices)} "
            "valid start states are available."
        )
    row_indices = np.sort(rng.choice(valid_indices, size=num_eval, replace=False))
    rows = dataset.get_row_data(row_indices)
    return row_indices, rows[col_name].tolist(), rows["step_idx"].tolist()


def make_world(
    variant: str,
    num_envs: int,
    max_episode_steps: int,
    background_video: str | None = None,
    background_seed: int = 0,
) -> swm.World:
    pre_wrappers = []
    if variant != "source":
        pre_wrappers.append(
            partial(
                ReacherRenderShiftWrapper,
                variant=variant,
                background_video=background_video,
                background_seed=background_seed,
            )
        )
    return swm.World(
        env_name="swm/ReacherDMControl-v0",
        num_envs=num_envs,
        max_episode_steps=max_episode_steps,
        task="qpos_match",
        image_shape=(224, 224),
        pre_wrappers=pre_wrappers,
    )


def shifted_render_state(
    init_state: dict,
    goal_state: dict,
    variant: str,
    episodes_idx,
    start_steps,
    goal_offset: int,
    background_video: str | None,
    background_seed: int,
    camera_step: float = 0.0,
) -> None:
    if variant == "source":
        return

    render_kwargs = {
        "background_video": background_video,
        "background_seed": background_seed,
        "background_episode_ids": episodes_idx,
    }

    seeds = init_state.get("seed")
    phase_offsets = None
    if seeds is not None and variant in DYNAMIC_CAMERA_VARIANTS:
        phase_offsets = np.asarray(
            [camera_phase_from_seed(seed) for seed in seeds], dtype=np.float64
        )
    elif variant in DYNAMIC_CAMERA_VARIANTS:
        phase_offsets = np.zeros(len(init_state["qpos"]), dtype=np.float64)

    render_variant_name = (
        "dynamic_camera" if variant in DYNAMIC_CAMERA_VARIANTS else variant
    )
    init_pixels = render_variant(
        {"qpos": init_state["qpos"], "qvel": init_state["qvel"]},
        render_variant_name,
        image_size=224,
        background_step_indices=start_steps,
        **render_kwargs,
        camera_steps=camera_step,
        camera_phase_offsets=phase_offsets,
    )
    goal_qvel = goal_state.get("goal_qvel")
    if goal_qvel is None:
        goal_qvel = np.zeros_like(init_state["qvel"])
    goal_pixels = render_variant(
        {"qpos": goal_state["goal_qpos"], "qvel": goal_qvel},
        render_variant_name,
        image_size=224,
        background_step_indices=np.asarray(start_steps) + goal_offset,
        **render_kwargs,
        camera_steps=camera_step,
        camera_phase_offsets=phase_offsets,
    )
    if variant in DYNAMIC_CAMERA_VARIANTS[1:]:
        background = (
            render_source_background(224)
            if variant == "dynamic_camera_oracle"
            else None
        )
        init_pixels = [
            canonicalize_dynamic_frame(
                frame,
                step=camera_step,
                phase_offset=float(phase_offsets[index]),
                source_background=background,
            )
            for index, frame in enumerate(init_pixels)
        ]
        goal_pixels = [
            canonicalize_dynamic_frame(
                frame,
                step=camera_step,
                phase_offset=float(phase_offsets[index]),
                source_background=background,
            )
            for index, frame in enumerate(goal_pixels)
        ]
    init_state["pixels"] = np.stack(init_pixels)
    goal_state["goal"] = np.stack(goal_pixels)


def extract_init_goal(dataset, episodes_idx, start_steps, goal_offset):
    ep_idx_arr = np.array(episodes_idx)
    start_arr = np.array(start_steps)
    data = dataset.load_chunk(
        ep_idx_arr, start_arr, start_arr + goal_offset + 1
    )

    init_lists: dict[str, list] = {}
    goal_lists: dict[str, list] = {}
    for ep in data:
        for col in dataset.column_names:
            if col.startswith("goal"):
                continue
            if col.startswith("pixels"):
                ep[col] = ep[col].permute(0, 2, 3, 1)
            val = ep[col]
            if not isinstance(val, (torch.Tensor, np.ndarray)):
                continue
            arr = val.numpy() if isinstance(val, torch.Tensor) else val
            init_lists.setdefault(col, []).append(arr[0])
            goal_lists.setdefault(col, []).append(arr[-1])

    init_state = {k: np.stack(v) for k, v in init_lists.items()}
    goal_state = {
        ("goal" if k == "pixels" else f"goal_{k}"): np.stack(v)
        for k, v in goal_lists.items()
    }
    return init_state, goal_state


def apply_callables(env, callables, init_state):
    for spec in callables:
        method = spec["method"]
        if not hasattr(env, method):
            continue
        prepared = {}
        for name, data in spec.get("args", {}).items():
            if data.get("in_dataset", True):
                key = data.get("value")
                if key in init_state:
                    prepared[name] = deepcopy(init_state[key])
            else:
                prepared[name] = data.get("value")
        getattr(env, method)(**prepared)


def evaluate_shifted_from_dataset(
    world,
    dataset,
    episodes_idx,
    start_steps,
    goal_offset,
    eval_budget,
    callables,
    variant,
    background_video,
    background_seed,
) -> dict:
    n = len(episodes_idx)
    assert n == world.num_envs

    init_state, goal_state = extract_init_goal(
        dataset, episodes_idx, start_steps, goal_offset
    )
    shifted_render_state(
        init_state,
        goal_state,
        variant,
        episodes_idx,
        start_steps,
        goal_offset,
        background_video,
        background_seed,
        camera_step=0.0,
    )

    reset_seed = init_state.get("seed")
    if reset_seed is not None:
        reset_seed = [int(value) for value in reset_seed]
    world.reset(seed=reset_seed)

    if callables:
        merged = {**init_state, **goal_state}
        for i in range(n):
            world_env = world.envs.envs[i]
            env_init = {k: v[i] for k, v in merged.items()}
            apply_callables(world_env.unwrapped, callables, env_init)
            current = world_env
            background_configured = False
            while current is not current.unwrapped:
                if isinstance(current, ReacherRenderShiftWrapper):
                    current.configure_background(episodes_idx[i], start_steps[i])
                    background_configured = True
                    break
                current = current.env
            if is_dcs_variant(variant) and not background_configured:
                raise RuntimeError("DCS render wrapper was not found in environment")

    shape_prefix = world.infos["pixels"].shape[:2]
    for src in (init_state, goal_state):
        for k, v in src.items():
            key = k
            if key in world.infos or key in goal_state:
                world.infos[key] = np.broadcast_to(
                    v[:, None, ...], shape_prefix + v.shape[1:]
                ).copy()

    goal_snapshot = {k: world.infos[k].copy() for k in goal_state}
    dynamic_goal_state = None
    dynamic_phase_offsets = None
    if variant in DYNAMIC_CAMERA_VARIANTS:
        goal_qvel = goal_state.get("goal_qvel")
        if goal_qvel is None:
            goal_qvel = np.zeros_like(init_state["qvel"])
        dynamic_goal_state = {
            "qpos": goal_state["goal_qpos"],
            "qvel": goal_qvel,
        }
        seeds = init_state.get("seed")
        if seeds is None:
            dynamic_phase_offsets = np.zeros(n, dtype=np.float64)
        else:
            dynamic_phase_offsets = np.asarray(
                [camera_phase_from_seed(seed) for seed in seeds], dtype=np.float64
            )
    dynamic_goal_env = (
        make_render_env("dynamic_camera") if dynamic_goal_state is not None else None
    )
    oracle_background = (
        render_source_background(224)
        if variant == "dynamic_camera_oracle"
        else None
    )
    camera_step = 0
    results = {
        "success_rate": 0.0,
        "episode_successes": np.zeros(n, dtype=bool),
        "seeds": init_state.get("seed"),
    }

    def on_step(active_world):
        nonlocal camera_step
        camera_step += 1
        if dynamic_goal_state is not None:
            goal_pixels = np.stack(
                render_variant(
                    dynamic_goal_state,
                    "dynamic_camera",
                    image_size=224,
                    camera_steps=camera_step,
                    camera_phase_offsets=dynamic_phase_offsets,
                    render_env=dynamic_goal_env,
                )
            )
            if variant in DYNAMIC_CAMERA_VARIANTS[1:]:
                goal_pixels = np.stack(
                    [
                        canonicalize_dynamic_frame(
                            frame,
                            step=camera_step,
                            phase_offset=float(dynamic_phase_offsets[index]),
                            source_background=oracle_background,
                        )
                        for index, frame in enumerate(goal_pixels)
                    ]
                )
            goal_snapshot["goal"] = np.broadcast_to(
                goal_pixels[:, None, ...],
                shape_prefix + goal_pixels.shape[1:],
            ).copy()
        active_world.infos.update(deepcopy(goal_snapshot))
        results["episode_successes"] |= active_world.terminateds

    try:
        world._run(max_steps=eval_budget, mode="wait", on_step=on_step)
    finally:
        if dynamic_goal_env is not None:
            dynamic_goal_env.close()
    results["success_rate"] = (
        float(results["episode_successes"].sum()) / n * 100.0
    )
    return results


def build_policy(args, dataset, model, device):
    processor = preprocessing.StandardScaler()
    action_data = dataset.get_col_data("action")
    action_data = action_data[~np.isnan(action_data).any(axis=1)]
    processor.fit(action_data)

    solver = swm.solver.CEMSolver(
        model=model,
        batch_size=1,
        num_samples=args.num_samples,
        var_scale=1.0,
        n_steps=args.cem_steps,
        topk=args.topk,
        device=device,
        seed=args.seed,
    )
    return swm.policy.WorldModelPolicy(
        solver=solver,
        config=swm.PlanConfig(
            horizon=args.horizon,
            receding_horizon=args.receding_horizon,
            action_block=args.action_block,
        ),
        process={"action": processor},
        transform={"pixels": img_transform(), "goal": img_transform()},
    )


def main() -> None:
    args = parse_args()
    cache_dir = resolve_cache_dir(args.cache_dir)
    dataset = swm.data.load_dataset(
        args.dataset_name, keys_to_cache=["action"], cache_dir=str(cache_dir)
    )
    row_indices, episodes_idx, start_steps = sample_eval_points(
        dataset,
        args.seed,
        args.num_eval,
        args.goal_offset_steps,
        episode_min=args.episode_min,
        episode_max=args.episode_max,
        exclude_episodes=args.exclude_episodes,
    )

    model = swm.wm.utils.load_pretrained(args.policy)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = model.to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    model, adapter_checkpoint = load_adapter_model(
        model, args.adapter_checkpoint, device
    )

    callables = [
        {
            "method": "set_state",
            "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}},
        },
        {
            "method": "set_target_qpos",
            "args": {"target_qpos": {"value": "goal_qpos"}},
        },
    ]

    report = {
        "policy": args.policy,
        "adapter_checkpoint": args.adapter_checkpoint,
        "adapter_report": (
            adapter_checkpoint.get("report") if adapter_checkpoint else None
        ),
        "dataset_name": args.dataset_name,
        "cache_dir": str(cache_dir),
        "device": device,
        "seed": args.seed,
        "num_eval": args.num_eval,
        "row_indices": [int(x) for x in row_indices],
        "episodes_idx": [int(x) for x in episodes_idx],
        "start_steps": [int(x) for x in start_steps],
        "background": (
            DavisBearBackground(
                args.background_video,
                dynamic="dcs_bear_dynamic" in args.variants,
                seed=args.background_seed,
            ).metadata()
            if any(is_dcs_variant(variant) for variant in args.variants)
            else None
        ),
        "protocol": {
            "episode_range": [args.episode_min, args.episode_max],
            "excluded_episodes": args.exclude_episodes,
            "goal_offset_steps": args.goal_offset_steps,
            "eval_budget": args.eval_budget,
            "horizon": args.horizon,
            "receding_horizon": args.receding_horizon,
            "action_block": args.action_block,
            "num_samples": args.num_samples,
            "cem_steps": args.cem_steps,
            "topk": args.topk,
        },
        "variants": {},
    }

    for variant in args.variants:
        if variant not in EVAL_VARIANTS:
            raise ValueError(
                f"Unknown variant {variant!r}; choose from {EVAL_VARIANTS}"
            )
        start_time = time.time()
        policy = build_policy(args, dataset, model, device)
        world = make_world(
            variant,
            args.num_eval,
            max_episode_steps=2 * args.eval_budget,
            background_video=args.background_video,
            background_seed=args.background_seed,
        )
        world.set_policy(policy)
        metrics = evaluate_shifted_from_dataset(
            world=world,
            dataset=dataset,
            episodes_idx=episodes_idx,
            start_steps=start_steps,
            goal_offset=args.goal_offset_steps,
            eval_budget=args.eval_budget,
            callables=callables,
            variant=variant,
            background_video=args.background_video,
            background_seed=args.background_seed,
        )
        world.close()
        elapsed = time.time() - start_time

        successes = metrics["episode_successes"]
        report["variants"][variant] = {
            "success_rate": float(metrics["success_rate"]),
            "successful_episodes": int(np.asarray(successes).sum()),
            "failed_episodes": int(args.num_eval - np.asarray(successes).sum()),
            "episode_successes": [bool(x) for x in successes],
            "wall_clock_seconds": elapsed,
        }
        print(
            f"{variant}: {metrics['success_rate']:.1f}% "
            f"({int(np.asarray(successes).sum())}/{args.num_eval}) "
            f"in {elapsed:.1f}s"
        )

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote: {out_path.resolve()}")
    print(OmegaConf.to_yaml(report["protocol"]))


if __name__ == "__main__":
    main()
