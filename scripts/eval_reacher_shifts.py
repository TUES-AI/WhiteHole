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

from scripts.visualize_reacher_shifts import (
    VARIANTS,
    color_grade,
    medium_variation_options,
    render_variant,
    set_hard_camera,
)
from scripts.reacher_conv_adapter import AdaptedLeWM, SmallConvAdapter
from scripts.reacher_movie_adapter import build_movie_encoder


class ReacherRenderShiftWrapper(gym.Wrapper):
    """Apply a visual-domain shift while leaving Reacher state/dynamics intact."""

    def __init__(self, env: gym.Env, variant: str):
        super().__init__(env)
        self.variant = variant

    def reset(self, *args, **kwargs):
        if self.variant == "medium_visual":
            options = dict(kwargs.get("options") or {})
            values = dict(options.get("variation_values") or {})
            values.update(medium_variation_options()["variation_values"])
            options["variation_values"] = values
            kwargs["options"] = options

        obs, info = self.env.reset(*args, **kwargs)
        if self.variant == "hard_camera":
            set_hard_camera(self.env.unwrapped)
        return obs, info

    def render(self, *args, **kwargs):
        if self.variant == "hard_camera":
            set_hard_camera(self.env.unwrapped)
        frame = self.env.render(*args, **kwargs)
        if self.variant == "medium_visual":
            frame = color_grade(frame)
        return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LeWM Reacher on source/medium/hard visual domains."
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

    adapter = SmallConvAdapter(
        channels=int(cfg.get("channels", 16)),
        depth=int(cfg.get("depth", 2)),
        residual_scale=float(cfg.get("residual_scale", 1.0)),
    ).to(device)
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


def sample_eval_points(dataset, seed: int, num_eval: int, goal_offset_steps: int):
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


def make_world(variant: str, num_envs: int, max_episode_steps: int) -> swm.World:
    pre_wrappers = []
    if variant != "source":
        pre_wrappers.append(
            partial(ReacherRenderShiftWrapper, variant=variant)
        )
    return swm.World(
        env_name="swm/ReacherDMControl-v0",
        num_envs=num_envs,
        max_episode_steps=max_episode_steps,
        task="qpos_match",
        image_shape=(224, 224),
        pre_wrappers=pre_wrappers,
    )


def shifted_render_state(init_state: dict, goal_state: dict, variant: str) -> None:
    if variant == "source":
        return

    init_pixels = render_variant(
        {"qpos": init_state["qpos"], "qvel": init_state["qvel"]},
        variant,
        image_size=224,
    )
    goal_qvel = goal_state.get("goal_qvel")
    if goal_qvel is None:
        goal_qvel = np.zeros_like(init_state["qvel"])
    goal_pixels = render_variant(
        {"qpos": goal_state["goal_qpos"], "qvel": goal_qvel},
        variant,
        image_size=224,
    )
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
) -> dict:
    n = len(episodes_idx)
    assert n == world.num_envs

    init_state, goal_state = extract_init_goal(
        dataset, episodes_idx, start_steps, goal_offset
    )
    shifted_render_state(init_state, goal_state, variant)

    world.reset(seed=init_state.get("seed"))

    if callables:
        merged = {**init_state, **goal_state}
        for i in range(n):
            env_init = {k: v[i] for k, v in merged.items()}
            apply_callables(world.envs.envs[i].unwrapped, callables, env_init)

    shape_prefix = world.infos["pixels"].shape[:2]
    for src in (init_state, goal_state):
        for k, v in src.items():
            key = k
            if key in world.infos or key in goal_state:
                world.infos[key] = np.broadcast_to(
                    v[:, None, ...], shape_prefix + v.shape[1:]
                ).copy()

    goal_snapshot = {k: world.infos[k].copy() for k in goal_state}
    results = {
        "success_rate": 0.0,
        "episode_successes": np.zeros(n, dtype=bool),
        "seeds": init_state.get("seed"),
    }

    def on_step(active_world):
        active_world.infos.update(deepcopy(goal_snapshot))
        results["episode_successes"] |= active_world.terminateds

    world._run(max_steps=eval_budget, mode="wait", on_step=on_step)
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
    dataset = swm.data.HDF5Dataset(
        args.dataset_name, keys_to_cache=["action"], cache_dir=cache_dir
    )
    row_indices, episodes_idx, start_steps = sample_eval_points(
        dataset, args.seed, args.num_eval, args.goal_offset_steps
    )

    model = swm.wm.utils.load_pretrained(args.policy)
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
        "protocol": {
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
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}; choose from {VARIANTS}")
        start_time = time.time()
        policy = build_policy(args, dataset, model, device)
        world = make_world(
            variant, args.num_eval, max_episode_steps=2 * args.eval_budget
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
