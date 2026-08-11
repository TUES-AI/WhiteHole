#!/usr/bin/env python3
"""Screen framewise image adapters on frozen V-JEPA 2.1 under an RBG shift."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import av
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import functional as TF

from scripts.jepa_visual_adapters import apply_visual_shift, build_image_adapter

MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
MASK_CONFIGS = (
    dict(spatial_scale=(0.15, 0.15), temporal_scale=(1.0, 1.0), aspect_ratio=(0.75, 1.5), num_blocks=8),
    dict(spatial_scale=(0.7, 0.7), temporal_scale=(1.0, 1.0), aspect_ratio=(0.75, 1.5), num_blocks=2),
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def clean_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key.replace("module.", "").replace("backbone.", ""): value
        for key, value in state.items()
    }


def decode_clip(path: Path, frames: int = 16, fps: float = 4.0, size: int = 384) -> torch.Tensor:
    """Decode one centered four-second clip as uint8 [C,T,H,W]."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        source_fps = float(stream.average_rate or 30.0)
        decoded = [torch.from_numpy(frame.to_ndarray(format="rgb24")) for frame in container.decode(stream)]
    if not decoded:
        raise RuntimeError(f"No video frames decoded from {path}")
    duration = len(decoded) / source_fps
    span = (frames - 1) / fps
    start = max(0.0, (duration - span) / 2.0)
    indices = [min(len(decoded) - 1, round((start + i / fps) * source_fps)) for i in range(frames)]
    clip = torch.stack([decoded[i] for i in indices]).permute(0, 3, 1, 2)
    height, width = clip.shape[-2:]
    resize_h = size if height <= width else round(height * size / width)
    resize_w = size if width <= height else round(width * size / height)
    clip = TF.resize(clip, [resize_h, resize_w], interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
    clip = TF.center_crop(clip, [size, size])
    return clip.permute(1, 0, 2, 3).contiguous().to(torch.uint8)


def cache_dataset(data_root: Path, cache_root: Path) -> tuple[list[str], dict[str, list[dict]]]:
    classes = sorted(path.name for path in (data_root / "train").iterdir() if path.is_dir())
    records: dict[str, list[dict]] = {split: [] for split in ("train", "val")}
    for split in records:
        for label, class_name in enumerate(classes):
            paths = sorted((data_root / split / class_name).glob("*.mp4"))
            for path in paths:
                relative = path.relative_to(data_root).with_suffix(".pt")
                cached = cache_root / relative
                if not cached.exists():
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(decode_clip(path), cached)
                records[split].append({"path": str(cached), "label": label, "class": class_name, "source": str(path)})
    return classes, records


def split_records(records: dict[str, list[dict]], classes: list[str]) -> dict[str, list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for split, rows in records.items():
        for row in rows:
            grouped.setdefault((split, row["class"]), []).append(row)
    output = {name: [] for name in ("probe", "adapt", "source_prior", "capacity", "eval")}
    for class_name in classes:
        train = grouped[("train", class_name)]
        val = grouped[("val", class_name)]
        if len(train) < 10 or len(val) < 10:
            raise ValueError("Expected at least 10 train and 10 validation videos per class")
        output["probe"] += train[:5]
        output["adapt"] += train[5:8]
        output["source_prior"] += train[8:10]
        output["capacity"] += val[:5]
        output["eval"] += val[5:10]
    return output


def load_batch(rows: list[dict], indices: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    clips = torch.stack([torch.load(rows[i]["path"], weights_only=True) for i in indices])
    labels = torch.tensor([rows[i]["label"] for i in indices], device=device)
    return clips.to(device=device, dtype=torch.float32).div_(255.0), labels


def batches(count: int, batch_size: int):
    for start in range(0, count, batch_size):
        yield list(range(start, min(start + batch_size, count)))


def shifted_video(clips: torch.Tensor) -> torch.Tensor:
    batch, channels, frames, height, width = clips.shape
    images = clips.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    indices = torch.arange(batch * frames, device=clips.device)
    images = apply_visual_shift(images, "rbg", indices)
    return images.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)


def adapt_video(adapter: nn.Module, clips: torch.Tensor) -> torch.Tensor:
    batch, channels, frames, height, width = clips.shape
    images = clips.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    images = adapter(images)
    return images.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)


def normalize(clips: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return ((clips - MEAN.to(clips.device)) / STD.to(clips.device)).to(dtype=dtype)


def build_models(repo: Path, student_checkpoint: Path, teacher_checkpoint: Path, device: torch.device):
    sys.path.insert(0, str(repo))
    from app.vjepa_2_1.models import predictor as predictor_module
    from app.vjepa_2_1.models import vision_transformer

    common = dict(
        img_size=(384, 384), patch_size=16, num_frames=16, tubelet_size=2,
        use_sdpa=True, use_silu=False, wide_silu=True, uniform_power=False,
        use_rope=True, img_temporal_dim_size=1, interpolate_rope=True,
        modality_embedding=True, n_output_distillation=1,
    )
    student = vision_transformer.vit_base(**common)
    downstream = vision_transformer.vit_base(**common)
    predictor = predictor_module.vit_predictor(
        img_size=(384, 384), patch_size=16, num_frames=16, tubelet_size=2,
        use_mask_tokens=True, embed_dim=768, predictor_embed_dim=384,
        teacher_embed_dim=1664, depth=12, num_heads=12, num_mask_tokens=8,
        use_rope=True, uniform_power=False, use_sdpa=True, use_silu=False,
        wide_silu=True, n_output_distillation=1, return_all_tokens=True,
        img_temporal_dim_size=1, modality_embedding=True,
    )
    student_state = torch.load(student_checkpoint, map_location="cpu", weights_only=True, mmap=True)
    student.load_state_dict(clean_state_dict(student_state["encoder"]), strict=True)
    downstream.load_state_dict(clean_state_dict(student_state["ema_encoder"]), strict=True)
    predictor.load_state_dict(clean_state_dict(student_state["predictor"]), strict=True)
    del student_state

    teacher = vision_transformer.vit_gigantic_xformers(**common)
    teacher_state = torch.load(teacher_checkpoint, map_location="cpu", weights_only=True, mmap=True)
    teacher.load_state_dict(clean_state_dict(teacher_state["target_encoder"]), strict=True)
    del teacher_state
    gc.collect()

    for model in (student, downstream, predictor, teacher):
        model.requires_grad_(False).eval().to(device=device)
    return student, downstream, predictor, teacher


def build_mask_generators(repo: Path):
    sys.path.insert(0, str(repo))
    from src.masks.multiseq_multiblock3d import _MaskGenerator

    return [
        _MaskGenerator(
            crop_size=(384, 384), num_frames=16, spatial_patch_size=(16, 16),
            temporal_patch_size=2, spatial_pred_mask_scale=config["spatial_scale"],
            temporal_pred_mask_scale=config["temporal_scale"], aspect_ratio=config["aspect_ratio"],
            npred=config["num_blocks"],
        )
        for config in MASK_CONFIGS
    ]


def sample_masks(batch_size: int, repo: Path, device: torch.device, generators=None):
    generators = build_mask_generators(repo) if generators is None else generators
    masks_enc, masks_pred = [], []
    for generator in generators:
        enc, pred = generator(batch_size)
        masks_enc.append(enc.to(device))
        masks_pred.append(pred.to(device))
    return masks_enc, masks_pred


def vjepa_loss(clips, student, predictor, teacher, masks_enc, masks_pred, repo: Path):
    sys.path.insert(0, str(repo))
    from app.vjepa_2_1.models.utils.masks_dist import compute_mask_distance

    videos = normalize(clips, torch.float32)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            targets = teacher(videos, training=True)
            targets = F.layer_norm(targets, (targets.shape[-1],))
        prediction_loss = videos.new_zeros((), dtype=torch.float32)
        context_loss = videos.new_zeros((), dtype=torch.float32)
        distances = compute_mask_distance([masks_pred], [masks_enc], 24, False)[0]
        for index, (mask_enc, mask_pred, distance) in enumerate(zip(masks_enc, masks_pred, distances)):
            context = student(videos, masks=mask_enc, training=True)
            prediction, predicted_context = predictor(context, mask_enc, mask_pred, mask_index=index, mod="video")
            target_prediction = torch.gather(targets, 1, mask_pred.unsqueeze(-1).expand(-1, -1, targets.shape[-1]))
            target_context = torch.gather(targets, 1, mask_enc.unsqueeze(-1).expand(-1, -1, targets.shape[-1]))
            prediction_loss = prediction_loss + (prediction.float() - target_prediction.float()).abs().mean()
            weighted = (predicted_context.float() - target_context.float()).abs() / distance.float().clamp_min(1e-6).unsqueeze(-1)
            context_loss = context_loss + weighted.mean()
    prediction_loss = prediction_loss / len(masks_enc)
    context_loss = context_loss / len(masks_enc)
    return prediction_loss + 0.5 * context_loss, prediction_loss, context_loss


@torch.no_grad()
def extract_features(rows, downstream, device, adapter=None, shift=False, batch_size=1):
    features, labels = [], []
    for ids in batches(len(rows), batch_size):
        clips, batch_labels = load_batch(rows, ids, device)
        if shift:
            clips = shifted_video(clips)
        if adapter is not None:
            clips = adapt_video(adapter, clips)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tokens = downstream(normalize(clips, torch.float32))
        features.append(tokens.float().mean(dim=1).cpu())
        labels.append(batch_labels.cpu())
    return torch.cat(features), torch.cat(labels)


def fit_probe(features: torch.Tensor, labels: torch.Tensor, classes: int, steps: int, seed: int) -> nn.Module:
    seed_everything(seed)
    features = F.normalize(features, dim=1)
    probe = nn.Linear(features.shape[1], classes)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=0.05, weight_decay=1e-3)
    for _ in range(steps):
        loss = F.cross_entropy(probe(features), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return probe.eval()


def probe_accuracy(features, labels, probe) -> float:
    with torch.no_grad():
        predictions = probe(F.normalize(features, dim=1)).argmax(dim=1)
    return float((predictions == labels).float().mean())


def stratified_indices(rows: list[dict], count: int) -> list[int]:
    by_label: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        by_label.setdefault(row["label"], []).append(index)
    selected = []
    depth = 0
    while len(selected) < min(count, len(rows)):
        for label in sorted(by_label):
            if depth < len(by_label[label]):
                selected.append(by_label[label][depth])
                if len(selected) == min(count, len(rows)):
                    break
        depth += 1
    return selected


@torch.no_grad()
def evaluate(rows, adapter, downstream, probe, student, predictor, teacher, masks, repo, device, loss_count):
    source_features, labels = extract_features(rows, downstream, device, adapter=adapter, shift=False)
    target_features, _ = extract_features(rows, downstream, device, adapter=adapter, shift=True)
    pixel_total = 0.0
    loss_total = pred_total = context_total = 0.0
    loss_indices = set(stratified_indices(rows, loss_count))
    for index in range(len(rows)):
        clean, _ = load_batch(rows, [index], device)
        shifted = shifted_video(clean)
        adapted = shifted if adapter is None else adapt_video(adapter, shifted)
        pixel_total += float(F.l1_loss(adapted, clean))
        if index in loss_indices:
            total, pred, context = vjepa_loss(adapted, student, predictor, teacher, *masks, repo)
            loss_total += float(total)
            pred_total += float(pred)
            context_total += float(context)
    denominator = len(loss_indices)
    return {
        "source_accuracy": probe_accuracy(source_features, labels, probe),
        "target_accuracy": probe_accuracy(target_features, labels, probe),
        "pixel_l1_to_source": pixel_total / len(rows),
        "vjepa_loss": loss_total / denominator,
        "prediction_loss": pred_total / denominator,
        "context_loss": context_total / denominator,
    }


def train_adapter(args, adapter_name, objective, rows, source_rows, student, predictor, teacher, repo, device):
    seed_everything(args.seed)
    adapter = build_image_adapter(adapter_name).to(device=device, dtype=torch.float32).train()
    seed_everything(args.seed)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.paired_lr if objective == "paired" else args.jepa_lr)
    mask_generators = build_mask_generators(repo)
    steps = args.paired_steps if objective == "paired" else args.jepa_steps
    batch_size = args.paired_batch_size if objective == "paired" else args.jepa_batch_size
    initial = {name: value.detach().clone() for name, value in adapter.named_parameters()}
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    last = {}
    for step in range(steps):
        ids = [((step * batch_size) + offset) % len(rows) for offset in range(batch_size)]
        clean, _ = load_batch(rows, ids, device)
        adapted = adapt_video(adapter, shifted_video(clean))
        if objective == "paired":
            primary = F.l1_loss(adapted, clean)
            source_ids = [((step * batch_size) + offset) % len(source_rows) for offset in range(batch_size)]
            source, _ = load_batch(source_rows, source_ids, device)
            identity = F.l1_loss(adapt_video(adapter, source), source)
            loss = primary + identity
            pred = context = primary.new_zeros(())
        else:
            current_masks = sample_masks(batch_size, repo, device, mask_generators)
            primary, pred, context = vjepa_loss(adapted, student, predictor, teacher, *current_masks, repo)
            identity = primary.new_zeros(())
            if objective == "source_prior":
                source_ids = [((step * batch_size) + offset) % len(source_rows) for offset in range(batch_size)]
                source, _ = load_batch(source_rows, source_ids, device)
                identity = F.l1_loss(adapt_video(adapter, source), source)
            loss = primary + identity
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite {objective} loss at step {step}: {loss}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 10.0)
        if not torch.isfinite(grad_norm) or grad_norm <= 0:
            raise RuntimeError(f"Invalid adapter gradient norm at step {step}: {grad_norm}")
        optimizer.step()
        last = {"loss": float(loss.detach()), "primary": float(primary.detach()), "identity": float(identity.detach()), "prediction": float(pred.detach()), "context": float(context.detach()), "gradient_norm": float(grad_norm)}
    max_update = max(float((value.detach() - initial[name]).abs().max()) for name, value in adapter.named_parameters())
    if max_update == 0:
        raise RuntimeError("Adapter parameters did not update")
    adapter.eval()
    return adapter, {
        "steps": steps, "last_train": last, "maximum_parameter_update": max_update,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "train_seconds": time.time() - start,
        "frozen_core_gradient_tensors": sum(p.grad is not None for model in (student, predictor, teacher) for p in model.parameters()),
    }


def write_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--vjepa-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapters", nargs="+", choices=("unet", "grid_color"), default=("unet", "grid_color"))
    parser.add_argument("--objectives", nargs="+", choices=("paired", "target", "source_prior"), default=("paired", "target", "source_prior"))
    parser.add_argument("--paired-steps", type=int, default=128)
    parser.add_argument("--jepa-steps", type=int, default=64)
    parser.add_argument("--paired-batch-size", type=int, default=1)
    parser.add_argument("--jepa-batch-size", type=int, default=1)
    parser.add_argument("--paired-lr", type=float, default=3e-4)
    parser.add_argument("--jepa-lr", type=float, default=1e-4)
    parser.add_argument("--probe-steps", type=int, default=1000)
    parser.add_argument("--eval-loss-clips", type=int, default=5)
    parser.add_argument("--capacity-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("V-JEPA 2.1 screen requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wall_start = time.time()
    device = torch.device("cuda")
    seed_everything(args.seed)
    classes, records = cache_dataset(args.data_root, args.cache_root)
    splits = split_records(records, classes)
    student, downstream, predictor, teacher = build_models(
        args.vjepa_repo, args.student_checkpoint, args.teacher_checkpoint, device
    )
    sentinel = {"student": next(student.parameters()).detach().clone(), "downstream": next(downstream.parameters()).detach().clone(), "predictor": next(predictor.parameters()).detach().clone(), "teacher": next(teacher.parameters()).detach().clone()}
    train_features, train_labels = extract_features(splits["probe"], downstream, device)
    probe = fit_probe(train_features, train_labels, len(classes), args.probe_steps, args.seed)
    eval_masks = sample_masks(1, args.vjepa_repo, device)
    report = {
        "protocol": "V-JEPA 2.1 ViT-B framewise visual adapter screen",
        "gpu": torch.cuda.get_device_name(), "seed": args.seed, "classes": classes,
        "split_counts": {key: len(value) for key, value in splits.items()},
        "student_checkpoint": {"path": str(args.student_checkpoint), "sha256": sha256(args.student_checkpoint)},
        "teacher_checkpoint": {"path": str(args.teacher_checkpoint), "sha256": sha256(args.teacher_checkpoint)},
        "shift": "RGB to RBG, identically on every frame", "fps": 4, "frames": 16, "crop_size": 384,
        "budgets": {"paired_steps": args.paired_steps, "jepa_steps": args.jepa_steps, "paired_batch_size": args.paired_batch_size, "jepa_batch_size": args.jepa_batch_size},
        "baseline": None, "rows": [], "skipped_rows": [],
    }
    report["baseline"] = evaluate(splits["eval"], None, downstream, probe, student, predictor, teacher, eval_masks, args.vjepa_repo, device, args.eval_loss_clips)
    report["source_probe_accuracy"] = report["baseline"]["source_accuracy"]
    baseline_capacity_pixel = 0.0
    for index in range(len(splits["capacity"])):
        clean, _ = load_batch(splits["capacity"], [index], device)
        baseline_capacity_pixel += float(F.l1_loss(shifted_video(clean), clean))
    baseline_capacity_pixel /= len(splits["capacity"])
    report["baseline"]["capacity_pixel_l1_to_source"] = baseline_capacity_pixel
    write_report(args.output_dir / "report.json", report)

    gates = {}
    for adapter_name in args.adapters:
        for objective in args.objectives:
            if objective != "paired" and not gates.get(adapter_name, False):
                report["skipped_rows"].append({"adapter": adapter_name, "objective": objective, "reason": "paired RBG capacity gate failed"})
                write_report(args.output_dir / "report.json", report)
                continue
            adapter, train_stats = train_adapter(args, adapter_name, objective, splits["adapt"], splits["source_prior"], student, predictor, teacher, args.vjepa_repo, device)
            row = {"adapter": adapter_name, "parameters": sum(p.numel() for p in adapter.parameters()), "objective": objective, **train_stats}
            row.update(evaluate(splits["eval"], adapter, downstream, probe, student, predictor, teacher, eval_masks, args.vjepa_repo, device, args.eval_loss_clips))
            if objective == "paired":
                capacity_pixel = 0.0
                with torch.no_grad():
                    for index in range(len(splits["capacity"])):
                        clean, _ = load_batch(splits["capacity"], [index], device)
                        capacity_pixel += float(F.l1_loss(adapt_video(adapter, shifted_video(clean)), clean))
                capacity_pixel /= len(splits["capacity"])
                recovery = 1.0 - capacity_pixel / baseline_capacity_pixel
                row["capacity_pixel_l1_to_source"] = capacity_pixel
                row["capacity_pixel_gap_recovery"] = recovery
                row["capacity_gate_pass"] = recovery >= args.capacity_threshold
                gates[adapter_name] = row["capacity_gate_pass"]
            torch.save(adapter.state_dict(), args.output_dir / f"{adapter_name}-{objective}.ckpt")
            report["rows"].append(row)
            write_report(args.output_dir / "report.json", report)
            del adapter
            gc.collect()
            torch.cuda.empty_cache()

    current = {"student": next(student.parameters()), "downstream": next(downstream.parameters()), "predictor": next(predictor.parameters()), "teacher": next(teacher.parameters())}
    report["core_sentinel_max_drift"] = max(float((current[name].detach() - sentinel[name]).abs().max()) for name in sentinel)
    report["wall_clock_seconds"] = time.time() - wall_start
    write_report(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
