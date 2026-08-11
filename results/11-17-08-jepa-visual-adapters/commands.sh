#!/usr/bin/env bash
# Primary fixed-domain I-JEPA run
PYTHONPATH=. /opt/venv/bin/python scripts/train_ijepa_visual_adapters.py \
  --data-root /proj/data/ijepa \
  --checkpoint /proj/checkpoints/IN1K-vit.h.14-300e.pth.tar \
  --ijepa-repo /proj/ijepa \
  --output-dir /proj/data/11-16-08-ijepa-fixed-visual-adapters \
  --probe-per-class 100 --adapt-per-class 100 \
  --source-prior-per-class 100 --capacity-val-per-class 20 \
  --eval-per-class 50 --probe-steps 500 --paired-steps 512 \
  --jepa-steps 256 --paired-batch-size 64 --jepa-batch-size 8 \
  --eval-batch-size 32 --eval-jepa-batches 20 \
  --num-workers 8

# Primary RBG V-JEPA run
PYTHONPATH=. /opt/venv/bin/python scripts/train_vjepa_visual_adapters.py \
  --data-root /proj/data/kinetics-mini \
  --cache-root /proj/data/kinetics-mini-cache \
  --student-checkpoint /proj/checkpoints/vjepa2_1_vitb_dist_vitG_384.pt \
  --teacher-checkpoint /proj/checkpoints/vjepa2_1_vitG_384.pt \
  --vjepa-repo /proj/vjepa2 \
  --output-dir /proj/data/11-17-08-vjepa-visual-adapters \
  --paired-steps 512 --jepa-steps 64 \
  --paired-batch-size 1 --jepa-batch-size 1 \
  --probe-steps 1000 --eval-loss-clips 5

# Faithful one-update V-JEPA target-loss CUDA smoke
PYTHONPATH=. /opt/venv/bin/python - <<'PY'
from argparse import Namespace
from pathlib import Path
import json
import torch
from scripts.train_vjepa_visual_adapters import build_models, cache_dataset, split_records, train_adapter
args = Namespace(seed=20260811, paired_lr=3e-4, jepa_lr=1e-4, paired_steps=1, jepa_steps=1, paired_batch_size=1, jepa_batch_size=1)
device = torch.device("cuda")
classes, records = cache_dataset(Path("/proj/data/kinetics-mini"), Path("/proj/data/kinetics-mini-cache"))
splits = split_records(records, classes)
student, downstream, predictor, teacher = build_models(Path("/proj/vjepa2"), Path("/proj/checkpoints/vjepa2_1_vitb_dist_vitG_384.pt"), Path("/proj/checkpoints/vjepa2_1_vitG_384.pt"), device)
adapter, stats = train_adapter(args, "unet", "target", splits["adapt"], splits["source_prior"], student, predictor, teacher, Path("/proj/vjepa2"), device)
result = {"status": "passed", "gpu": torch.cuda.get_device_name(), "seed": args.seed, "adapter": "unet", "shift": "rbg", "objective": "target", **stats}
Path("/proj/data/vjepa-target-gradient-smoke.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY
