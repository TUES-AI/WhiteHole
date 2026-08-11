"""Screen small image adapters on structured I-JEPA visual shifts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from scripts.jepa_visual_adapters import (
    SHIFT_NAMES,
    apply_visual_shift,
    build_image_adapter,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ijepa-repo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--adapters", nargs="+", default=["unet", "grid_color"])
    parser.add_argument(
        "--shifts", nargs="+", default=["rbg", "affine", "composed"]
    )
    parser.add_argument(
        "--objectives", nargs="+", default=["paired", "target", "source_prior"]
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--probe-per-class", type=int, default=100)
    parser.add_argument("--adapt-per-class", type=int, default=100)
    parser.add_argument("--source-prior-per-class", type=int, default=100)
    parser.add_argument("--capacity-val-per-class", type=int, default=20)
    parser.add_argument("--eval-per-class", type=int, default=50)
    parser.add_argument("--probe-steps", type=int, default=500)
    parser.add_argument("--paired-steps", type=int, default=256)
    parser.add_argument("--jepa-steps", type=int, default=128)
    parser.add_argument("--paired-batch-size", type=int, default=32)
    parser.add_argument("--jepa-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--paired-lr", type=float, default=3e-4)
    parser.add_argument("--source-prior-weight", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-jepa-batches", type=int, default=8)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


class ImageNetDerivedDataset(Dataset):
    """Combine Imagenette and Imagewoof without merging their class labels."""

    def __init__(self, root: str | Path, split: str):
        root = Path(root)
        class_dirs = []
        for dataset_name in ("imagenette2", "imagewoof2"):
            split_root = root / dataset_name / split
            if not split_root.is_dir():
                raise FileNotFoundError(split_root)
            class_dirs.extend(sorted(path for path in split_root.iterdir() if path.is_dir()))
        self.classes = [path.name for path in class_dirs]
        self.samples = []
        extensions = {".jpg", ".jpeg", ".png", ".webp"}
        for label, class_dir in enumerate(class_dirs):
            for path in sorted(class_dir.rglob("*")):
                if path.suffix.lower() in extensions and not path.name.startswith("._"):
                    self.samples.append((path, label))
        self.transform = transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = self.transform(image.convert("RGB"))
        return image, label, index


def balanced_indices(
    dataset: ImageNetDerivedDataset, count_per_class: int, seed: int
) -> list[int]:
    by_class = defaultdict(list)
    for index, (_path, label) in enumerate(dataset.samples):
        by_class[label].append(index)
    generator = torch.Generator().manual_seed(seed)
    selected_by_class = []
    for label in range(len(dataset.classes)):
        values = torch.tensor(by_class[label])
        if len(values) < count_per_class:
            raise ValueError(
                f"Class {dataset.classes[label]} has {len(values)} images, "
                f"fewer than requested {count_per_class}."
            )
        order = torch.randperm(len(values), generator=generator)
        selected_by_class.append(values[order[:count_per_class]].tolist())
    return torch.tensor(selected_by_class).T.flatten().tolist()


def split_train_indices(
    dataset,
    probe_per_class,
    adapt_per_class,
    source_prior_per_class,
    capacity_val_per_class,
    seed,
):
    by_class = defaultdict(list)
    for index, (_path, label) in enumerate(dataset.samples):
        by_class[label].append(index)
    generator = torch.Generator().manual_seed(seed)
    probe, adapt, source_prior, capacity_val = [], [], [], []
    for label in range(len(dataset.classes)):
        values = torch.tensor(by_class[label])
        required = (
            probe_per_class
            + adapt_per_class
            + source_prior_per_class
            + capacity_val_per_class
        )
        if len(values) < required:
            raise ValueError(f"Class {dataset.classes[label]} has only {len(values)} images")
        values = values[torch.randperm(len(values), generator=generator)]
        probe.extend(values[:probe_per_class].tolist())
        adapt_end = probe_per_class + adapt_per_class
        adapt.extend(values[probe_per_class:adapt_end].tolist())
        source_end = adapt_end + source_prior_per_class
        source_prior.extend(values[adapt_end:source_end].tolist())
        capacity_val.extend(values[source_end:required].tolist())
    return probe, adapt, source_prior, capacity_val


def make_loader(dataset, indices, batch_size, shuffle, workers, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        drop_last=shuffle,
        generator=generator,
    )


def cycle_loader(loader):
    while True:
        yield from loader


def clean_state_dict(state: dict) -> dict:
    return {key.removeprefix("module."): value for key, value in state.items()}


def load_ijepa(args, device, dtype):
    sys.path.insert(0, str(Path(args.ijepa_repo).resolve()))
    helper = importlib.import_module("src.helper")
    encoder, predictor = helper.init_model(
        device="cpu",
        patch_size=14,
        model_name="vit_huge",
        crop_size=224,
        pred_depth=12,
        pred_emb_dim=384,
    )
    target_encoder = copy.deepcopy(encoder)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True, mmap=True
    )
    encoder.load_state_dict(clean_state_dict(checkpoint["encoder"]), strict=True)
    predictor.load_state_dict(clean_state_dict(checkpoint["predictor"]), strict=True)
    target_encoder.load_state_dict(
        clean_state_dict(checkpoint["target_encoder"]), strict=True
    )
    metadata = {
        "epoch": int(checkpoint["epoch"]),
        "checkpoint_loss": float(checkpoint["loss"]),
    }
    del checkpoint
    modules = (encoder, predictor, target_encoder)
    for module in modules:
        module.requires_grad_(False)
        module.eval()
        module.to(device=device, dtype=dtype)
    mask_module = importlib.import_module("src.masks.multiblock")
    tensor_module = importlib.import_module("src.utils.tensors")
    mask_collator = mask_module.MaskCollator(
        input_size=224,
        patch_size=14,
        pred_mask_scale=(0.15, 0.2),
        enc_mask_scale=(0.85, 1.0),
        aspect_ratio=(0.75, 1.5),
        nenc=1,
        npred=4,
        allow_overlap=False,
        min_keep=10,
    )
    return encoder, predictor, target_encoder, mask_collator, tensor_module, metadata


def normalize(images: torch.Tensor) -> torch.Tensor:
    mean = images.new_tensor(IMAGENET_MEAN)[None, :, None, None]
    std = images.new_tensor(IMAGENET_STD)[None, :, None, None]
    return (images - mean) / std


def sample_masks(mask_collator, batch_size, device):
    _unused, masks_encoder, masks_predictor = mask_collator(
        [torch.tensor(0) for _ in range(batch_size)]
    )
    return (
        [mask.to(device) for mask in masks_encoder],
        [mask.to(device) for mask in masks_predictor],
    )


def ijepa_loss(
    adapted,
    encoder,
    predictor,
    target_encoder,
    mask_collator,
    tensor_module,
    device,
    fixed_masks=None,
):
    core_dtype = next(encoder.parameters()).dtype
    images = normalize(adapted).to(dtype=core_dtype)
    masks_encoder, masks_predictor = (
        fixed_masks
        if fixed_masks is not None
        else sample_masks(mask_collator, len(images), device)
    )
    with torch.no_grad():
        target = target_encoder(images.detach())
        target = F.layer_norm(target, (target.shape[-1],))
        target = tensor_module.apply_masks(target, masks_predictor)
        target = tensor_module.repeat_interleave_batch(
            target, len(images), repeat=len(masks_encoder)
        )
    context = encoder(images, masks_encoder)
    prediction = predictor(context, masks_encoder, masks_predictor)
    return F.smooth_l1_loss(prediction, target)


@torch.no_grad()
def extract_features(loader, target_encoder, adapter, shift, device, dtype):
    features, labels = [], []
    for images, batch_labels, indices in loader:
        shifted = apply_visual_shift(images, shift, indices).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        if adapter is not None:
            shifted = adapter(shifted)
        core_dtype = next(target_encoder.parameters()).dtype
        embedding = target_encoder(normalize(shifted).to(core_dtype)).mean(dim=1)
        features.append(embedding.float().cpu())
        labels.append(batch_labels)
    return torch.cat(features), torch.cat(labels)


def train_probe(features, labels, num_classes, steps, seed, device):
    torch.manual_seed(seed)
    features = F.normalize(features.to(device), dim=-1)
    labels = labels.to(device)
    probe = nn.Linear(features.shape[-1], num_classes).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=3e-3, weight_decay=1e-4)
    batch_size = min(512, len(features))
    for _ in range(steps):
        indices = torch.randint(len(features), (batch_size,), device=device)
        loss = F.cross_entropy(probe(features[indices]), labels[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    probe.eval().requires_grad_(False)
    return probe


def probe_accuracy(probe, features, labels, device):
    with torch.no_grad():
        logits = probe(F.normalize(features.to(device), dim=-1))
        return float((logits.argmax(dim=-1).cpu() == labels).float().mean())


def train_adapter(
    args,
    adapter_name,
    shift,
    objective,
    train_dataset,
    adapt_indices,
    source_prior_indices,
    encoder,
    predictor,
    target_encoder,
    mask_collator,
    tensor_module,
    device,
):
    row_seed = (
        args.seed
        + 100 * args.shifts.index(shift)
        + 10 * args.objectives.index(objective)
    )
    torch.manual_seed(row_seed)
    with mask_collator._itr_counter.get_lock():
        mask_collator._itr_counter.value = -1
    adapter = build_image_adapter(adapter_name).to(device=device).train()
    parameters = sum(parameter.numel() for parameter in adapter.parameters())
    initial_parameters = [parameter.detach().clone() for parameter in adapter.parameters()]
    paired = objective == "paired"
    steps = args.paired_steps if paired else args.jepa_steps
    lr = args.paired_lr if paired else args.lr
    batch_size = args.paired_batch_size if paired else args.jepa_batch_size
    target_loader = make_loader(
        train_dataset,
        adapt_indices,
        batch_size,
        True,
        args.num_workers,
        row_seed,
    )
    source_loader = make_loader(
        train_dataset,
        source_prior_indices,
        args.jepa_batch_size,
        True,
        args.num_workers,
        row_seed + 1,
    )
    target_iterator = cycle_loader(target_loader)
    source_iterator = cycle_loader(source_loader)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
    # Architecture initialization consumes different amounts of randomness.
    # Reset before mask sampling so architectures see identical training masks.
    torch.manual_seed(row_seed + 5000)
    history = []
    frozen = list(encoder.parameters()) + list(predictor.parameters()) + list(
        target_encoder.parameters()
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.time()
    first_gradient_norm = None
    for step in range(1, steps + 1):
        clean, _labels, indices = next(target_iterator)
        shifted = apply_visual_shift(clean, shift, indices).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        clean = clean.to(device=device, dtype=torch.float32, non_blocking=True)
        adapted = adapter(shifted)
        if not torch.isfinite(adapted).all():
            raise FloatingPointError(f"Non-finite adapter output at step {step}")
        if paired:
            primary = F.l1_loss(adapted, clean)
            source, _source_labels, _source_indices = next(source_iterator)
            source = source.to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            source_identity = F.l1_loss(adapter(source), source)
            loss = primary + args.source_prior_weight * source_identity
        else:
            primary = ijepa_loss(
                adapted,
                encoder,
                predictor,
                target_encoder,
                mask_collator,
                tensor_module,
                device,
            )
            source_identity = primary.new_zeros(())
            if objective == "source_prior":
                source, _source_labels, _source_indices = next(source_iterator)
                source = source.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                source_identity = F.l1_loss(adapter(source), source)
            loss = primary + args.source_prior_weight * source_identity
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}: {loss}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_squared = sum(
            float(parameter.grad.detach().float().square().sum())
            for parameter in adapter.parameters()
            if parameter.grad is not None
        )
        gradient_norm = math.sqrt(gradient_squared)
        if not math.isfinite(gradient_norm) or gradient_norm == 0:
            raise RuntimeError(f"Invalid adapter gradient norm: {gradient_norm}")
        if first_gradient_norm is None:
            first_gradient_norm = gradient_norm
        optimizer.step()
        if step == 1 or step == steps or step % max(1, steps // 8) == 0:
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "primary": float(primary.detach()),
                "source_identity": float(source_identity.detach()),
                "gradient_norm": gradient_norm,
            }
            history.append(row)
            print(adapter_name, shift, objective, row, flush=True)
    update_max = max(
        float((parameter.detach() - initial).abs().max())
        for parameter, initial in zip(adapter.parameters(), initial_parameters)
    )
    if not math.isfinite(update_max) or update_max == 0:
        raise RuntimeError(f"Adapter parameters did not update: {update_max}")
    frozen_grad_tensors = sum(parameter.grad is not None for parameter in frozen)
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    adapter.eval().requires_grad_(False)
    return adapter, {
        "adapter": adapter_name,
        "shift": shift,
        "objective": objective,
        "trainable_parameters": parameters,
        "steps": steps,
        "learning_rate": lr,
        "history": history,
        "first_gradient_norm": first_gradient_norm,
        "parameter_update_max": update_max,
        "frozen_core_gradient_tensors": frozen_grad_tensors,
        "peak_cuda_memory_bytes": peak_memory,
        "wall_clock_seconds": time.time() - start_time,
    }


@torch.no_grad()
def paired_pixel_l1(loader, adapter, shift, device, dtype):
    total, count = 0.0, 0
    for images, _labels, indices in loader:
        shifted = apply_visual_shift(images, shift, indices)
        if adapter is None:
            adapted = shifted
        else:
            shifted = shifted.to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            images = images.to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            adapted = adapter(shifted)
        batch_size = len(images)
        batch_l1 = F.l1_loss(adapted.float(), images.float())
        total += float(batch_l1) * batch_size
        count += batch_size
    return total / count


def make_fixed_eval_masks(loader, mask_collator, device, max_batches, seed):
    masks = []
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        with mask_collator._itr_counter.get_lock():
            mask_collator._itr_counter.value = -1
        for batch_index, (images, _labels, _indices) in enumerate(loader):
            if batch_index >= max_batches:
                break
            masks.append(sample_masks(mask_collator, len(images), device))
    return masks


@torch.no_grad()
def average_ijepa_loss(
    loader,
    adapter,
    shift,
    encoder,
    predictor,
    target_encoder,
    mask_collator,
    tensor_module,
    device,
    dtype,
    fixed_eval_masks,
):
    values = []
    for batch_index, (images, _labels, indices) in enumerate(loader):
        if batch_index >= len(fixed_eval_masks):
            break
        shifted = apply_visual_shift(images, shift, indices).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        adapted = shifted if adapter is None else adapter(shifted)
        values.append(
            float(
                ijepa_loss(
                    adapted,
                    encoder,
                    predictor,
                    target_encoder,
                    mask_collator,
                    tensor_module,
                    device,
                    fixed_masks=fixed_eval_masks[batch_index],
                )
            )
        )
    return sum(values) / len(values)


@torch.no_grad()
def adapter_output_stats(loader, adapter, shift, device):
    minimum, maximum, finite = float("inf"), float("-inf"), True
    for images, _labels, indices in loader:
        shifted = apply_visual_shift(images, shift, indices)
        output = (
            shifted
            if adapter is None
            else adapter(
                shifted.to(device=device, dtype=torch.float32, non_blocking=True)
            )
        )
        minimum = min(minimum, float(output.min()))
        maximum = max(maximum, float(output.max()))
        finite = finite and bool(torch.isfinite(output).all())
    return {"minimum": minimum, "maximum": maximum, "all_finite": finite}


def capture_core_sentinels(modules):
    sentinels = {}
    for module_name, module in modules.items():
        parameters = list(module.named_parameters())
        for parameter_name, parameter in (parameters[0], parameters[-1]):
            key = f"{module_name}.{parameter_name}"
            sentinels[key] = parameter.detach().cpu().clone()
    return sentinels


def core_sentinel_drift(modules, sentinels):
    drift = 0.0
    for module_name, module in modules.items():
        parameters = dict(module.named_parameters())
        for key, reference in sentinels.items():
            prefix = f"{module_name}."
            if key.startswith(prefix):
                current = parameters[key.removeprefix(prefix)].detach().cpu()
                drift = max(drift, float((current - reference).abs().max()))
    return drift


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    invalid_shifts = set(args.shifts) - set(SHIFT_NAMES)
    if invalid_shifts:
        raise ValueError(f"Unknown shifts: {sorted(invalid_shifts)}")
    device = torch.device(
        args.device
        or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    )
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = ImageNetDerivedDataset(args.data_root, "train")
    eval_dataset = ImageNetDerivedDataset(args.data_root, "val")
    (
        probe_indices,
        adapt_indices,
        source_prior_indices,
        capacity_val_indices,
    ) = split_train_indices(
        train_dataset,
        args.probe_per_class,
        args.adapt_per_class,
        args.source_prior_per_class,
        args.capacity_val_per_class,
        args.seed,
    )
    assert not (set(probe_indices) & set(adapt_indices))
    assert not (set(probe_indices) & set(source_prior_indices))
    assert not (set(adapt_indices) & set(source_prior_indices))
    assert not (set(capacity_val_indices) & set(probe_indices))
    assert not (set(capacity_val_indices) & set(adapt_indices))
    assert not (set(capacity_val_indices) & set(source_prior_indices))
    eval_indices = balanced_indices(eval_dataset, args.eval_per_class, args.seed + 1)
    probe_loader = make_loader(
        train_dataset,
        probe_indices,
        args.eval_batch_size,
        False,
        args.num_workers,
        args.seed,
    )
    eval_loader = make_loader(
        eval_dataset,
        eval_indices,
        args.eval_batch_size,
        False,
        args.num_workers,
        args.seed,
    )
    capacity_val_loader = make_loader(
        train_dataset,
        capacity_val_indices,
        args.eval_batch_size,
        False,
        args.num_workers,
        args.seed,
    )

    encoder, predictor, target_encoder, mask_collator, tensor_module, metadata = load_ijepa(
        args, device, dtype
    )
    core_modules = {
        "encoder": encoder,
        "predictor": predictor,
        "target_encoder": target_encoder,
    }
    core_sentinels = capture_core_sentinels(core_modules)
    fixed_eval_masks = make_fixed_eval_masks(
        eval_loader,
        mask_collator,
        device,
        args.eval_jepa_batches,
        args.seed + 999,
    )
    source_train_features, source_train_labels = extract_features(
        probe_loader, target_encoder, None, "source", device, dtype
    )
    probe = train_probe(
        source_train_features,
        source_train_labels,
        len(train_dataset.classes),
        args.probe_steps,
        args.seed,
        device,
    )
    source_eval_features, source_eval_labels = extract_features(
        eval_loader, target_encoder, None, "source", device, dtype
    )
    source_accuracy = probe_accuracy(
        probe, source_eval_features, source_eval_labels, device
    )
    print(f"source_accuracy={source_accuracy:.4f}", flush=True)

    report = {
        "experiment": "structured_visual_adaptation_of_frozen_ijepa",
        "method_status": "single-seed exploratory screening",
        "device": str(device),
        "core_dtype": str(dtype),
        "adapter_dtype": str(torch.float32),
        "seed": args.seed,
        "classes": train_dataset.classes,
        "data": {
            "dataset": "Imagenette + Imagewoof (20 ImageNet-1K classes)",
            "root": str(Path(args.data_root).resolve()),
            "probe_images": len(probe_indices),
            "target_adaptation_images": len(adapt_indices),
            "disjoint_source_prior_images": len(source_prior_indices),
            "capacity_validation_images": len(capacity_val_indices),
            "evaluation_images": len(eval_indices),
        },
        "checkpoint": {
            "path": str(Path(args.checkpoint).resolve()),
            "sha256": checkpoint_sha256(args.checkpoint),
            **metadata,
        },
        "protocol": {
            "shifts": args.shifts,
            "affine_target_domain_degrees": {
                "rotation": 30.0,
                "shear_x": 18.0,
                "shear_y": 8.0,
            },
            "adapters": args.adapters,
            "objectives": args.objectives,
            "source_prior": "pixel identity on a disjoint, unpaired clean-source subset",
            "paired_gate": {
                "split": "disjoint capacity-validation subset from training data",
                "minimum_pixel_gap_recovery": 0.5,
            },
            "evaluation_masks": "fixed and shared by every row",
            "selection": "no shifted evaluation labels used for training",
        },
        "source_accuracy": source_accuracy,
        "baselines": {},
        "rows": [],
        "skipped_rows": [],
    }

    def write_report():
        report["core_sentinel_max_drift"] = core_sentinel_drift(
            core_modules, core_sentinels
        )
        report["wall_clock_seconds"] = sum(
            row["wall_clock_seconds"] for row in report["rows"]
        )
        (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    sample_images, _sample_labels, sample_indices = next(iter(eval_loader))
    visual_columns = [sample_images[:8]]
    for shift in args.shifts:
        shifted_features, labels = extract_features(
            eval_loader, target_encoder, None, shift, device, dtype
        )
        shifted_accuracy = probe_accuracy(probe, shifted_features, labels, device)
        baseline_l1 = paired_pixel_l1(eval_loader, None, shift, device, dtype)
        capacity_baseline_l1 = paired_pixel_l1(
            capacity_val_loader, None, shift, device, dtype
        )
        baseline_jepa = average_ijepa_loss(
            eval_loader,
            None,
            shift,
            encoder,
            predictor,
            target_encoder,
            mask_collator,
            tensor_module,
            device,
            dtype,
            fixed_eval_masks,
        )
        report["baselines"][shift] = {
            "accuracy": shifted_accuracy,
            "pixel_l1_to_source": baseline_l1,
            "capacity_val_pixel_l1_to_source": capacity_baseline_l1,
            "ijepa_loss": baseline_jepa,
            "output": adapter_output_stats(eval_loader, None, shift, device),
        }
        visual_columns.append(
            apply_visual_shift(
                sample_images[:8],
                shift,
                sample_indices[:8],
            )
        )
        print(
            f"baseline {shift}: accuracy={shifted_accuracy:.4f} "
            f"pixel_l1={baseline_l1:.4f} jepa={baseline_jepa:.4f}",
            flush=True,
        )
        write_report()

    objective_order = [
        objective
        for objective in ("paired", "target", "source_prior")
        if objective in args.objectives
    ]
    for adapter_name in args.adapters:
        for shift in args.shifts:
            capacity_pass = "paired" not in objective_order
            for objective in objective_order:
                if objective != "paired" and not capacity_pass:
                    report["skipped_rows"].append(
                        {
                            "adapter": adapter_name,
                            "shift": shift,
                            "objective": objective,
                            "reason": "paired capacity gate failed",
                        }
                    )
                    write_report()
                    continue
                adapter, row = train_adapter(
                    args,
                    adapter_name,
                    shift,
                    objective,
                    train_dataset,
                    adapt_indices,
                    source_prior_indices,
                    encoder,
                    predictor,
                    target_encoder,
                    mask_collator,
                    tensor_module,
                    device,
                )
                clean_features, clean_labels = extract_features(
                    eval_loader, target_encoder, adapter, "source", device, dtype
                )
                shifted_features, shifted_labels = extract_features(
                    eval_loader, target_encoder, adapter, shift, device, dtype
                )
                row.update(
                    {
                        "source_accuracy": probe_accuracy(
                            probe, clean_features, clean_labels, device
                        ),
                        "target_accuracy": probe_accuracy(
                            probe, shifted_features, shifted_labels, device
                        ),
                        "pixel_l1_to_source": paired_pixel_l1(
                            eval_loader, adapter, shift, device, dtype
                        ),
                        "ijepa_loss": average_ijepa_loss(
                            eval_loader,
                            adapter,
                            shift,
                            encoder,
                            predictor,
                            target_encoder,
                            mask_collator,
                            tensor_module,
                            device,
                            dtype,
                            fixed_eval_masks,
                        ),
                        "output": adapter_output_stats(
                            eval_loader, adapter, shift, device
                        ),
                    }
                )
                baseline = report["baselines"][shift]
                accuracy_gap = source_accuracy - baseline["accuracy"]
                row["gap_recovery"] = (
                    (row["target_accuracy"] - baseline["accuracy"]) / accuracy_gap
                    if accuracy_gap > 0
                    else None
                )
                row["pixel_gap_recovery"] = 1.0 - (
                    row["pixel_l1_to_source"] / baseline["pixel_l1_to_source"]
                )
                if objective == "paired":
                    capacity_l1 = paired_pixel_l1(
                        capacity_val_loader, adapter, shift, device, dtype
                    )
                    row["capacity_val_pixel_l1_to_source"] = capacity_l1
                    row["capacity_val_pixel_gap_recovery"] = 1.0 - (
                        capacity_l1 / baseline["capacity_val_pixel_l1_to_source"]
                    )
                    capacity_pass = row["capacity_val_pixel_gap_recovery"] >= 0.5
                    row["capacity_gate_pass"] = capacity_pass
                report["rows"].append(row)
                checkpoint = {
                    "adapter_state_dict": {
                        key: value.detach().cpu()
                        for key, value in adapter.state_dict().items()
                    },
                    "adapter_config": {"architecture": adapter_name},
                    "row": row,
                }
                torch.save(
                    checkpoint,
                    output_dir / f"{adapter_name}_{shift}_{objective}.ckpt",
                )
                print("RESULT", json.dumps(row), flush=True)
                write_report()
                del adapter
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    save_image(
        make_grid(torch.cat(visual_columns), nrow=8, padding=2),
        output_dir / "shift_examples.png",
    )
    write_report()
    if report["core_sentinel_max_drift"] != 0:
        raise RuntimeError(
            f"Frozen I-JEPA core changed: {report['core_sentinel_max_drift']}"
        )
    print(f"wrote {output_dir / 'report.json'}")


if __name__ == "__main__":
    main()
