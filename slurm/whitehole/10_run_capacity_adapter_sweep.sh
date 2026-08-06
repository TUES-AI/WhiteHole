#!/bin/bash
# Run higher-capacity residual adapter tests after the delta/diagonal ablations.
#
# Submit from /valhalla/projects/bg-eng-01/WhiteHole:
#   sbatch slurm/whitehole/10_run_capacity_adapter_sweep.sh

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=wh_cap_sweep
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o /valhalla/projects/bg-eng-01/WhiteHole/logs/wh_cap_sweep.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/WhiteHole/logs/wh_cap_sweep.%j.err

set -euo pipefail

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/WhiteHole"
SWEEP_SCRIPT="${SWEEP_SCRIPT:-slurm/whitehole/07_run_delta_pair_anchor_sweep.sh}"
SUMMARY_JSON="${SUMMARY_JSON:-outputs/eval/capacity_adapters/capacity_summary.json}"
SUMMARY_CSV="${SUMMARY_CSV:-outputs/eval/capacity_adapters/capacity_summary.csv}"

[ -d "${PROJECT_DIR}" ] || { echo "Missing project dir: ${PROJECT_DIR}"; exit 1; }

cd "${PROJECT_DIR}"
mkdir -p logs outputs/eval/capacity_adapters

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting capacity adapter sweep"
echo "  summary_json=${SUMMARY_JSON}"
echo "  summary_csv=${SUMMARY_CSV}"

echo ""
echo "===================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Low-rank residual adapter"
echo "===================================================="

CONFIG="configs/adaptation/two_rooms_medium_lowrank.yaml" \
PAIR_ALIGNMENT_WEIGHTS="0.3 1.0 3.0" \
EPOCHS=3 \
LR=0.0003 \
WEIGHT_DECAY=0.000001 \
MAX_TRAIN_BATCHES_PER_EPOCH=1000 \
DELTA_INIT_BATCHES=64 \
SOURCE_SCALE_BATCHES=64 \
VAL_BATCHES=64 \
LOCAL_ISOMETRY_WEIGHT=0.01 \
LOCAL_ISOMETRY_SAMPLES=128 \
SOURCE_IDENTITY_WEIGHT=0.1 \
VARIANCE_ALIGNMENT_WEIGHT=0.1 \
COVARIANCE_ALIGNMENT_WEIGHT=0.01 \
COVARIANCE_SAMPLES=512 \
OUTPUT_ROOT="outputs/adaptation/capacity_adapters/lowrank_r32" \
EVAL_OUTPUT_ROOT="outputs/eval/capacity_adapters/lowrank_r32" \
RUN_PREFIX="two_rooms_medium_lowrank_pairw" \
RUN_SUFFIX="3ep" \
bash "${SWEEP_SCRIPT}"

echo ""
echo "===================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Residual MLP adapter"
echo "===================================================="

CONFIG="configs/adaptation/two_rooms_medium_residual_mlp.yaml" \
PAIR_ALIGNMENT_WEIGHTS="0.3 1.0 3.0" \
EPOCHS=3 \
LR=0.0003 \
WEIGHT_DECAY=0.000001 \
MAX_TRAIN_BATCHES_PER_EPOCH=500 \
DELTA_INIT_BATCHES=64 \
SOURCE_SCALE_BATCHES=64 \
VAL_BATCHES=64 \
LOCAL_ISOMETRY_WEIGHT=0.005 \
LOCAL_ISOMETRY_SAMPLES=64 \
SOURCE_IDENTITY_WEIGHT=0.1 \
VARIANCE_ALIGNMENT_WEIGHT=0.1 \
COVARIANCE_ALIGNMENT_WEIGHT=0.01 \
COVARIANCE_SAMPLES=512 \
OUTPUT_ROOT="outputs/adaptation/capacity_adapters/mlp_h256" \
EVAL_OUTPUT_ROOT="outputs/eval/capacity_adapters/mlp_h256" \
RUN_PREFIX="two_rooms_medium_mlp_pairw" \
RUN_SUFFIX="3ep" \
bash "${SWEEP_SCRIPT}"

echo ""
echo "===================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Summarizing capacity sweep"
echo "===================================================="

python - <<PY
import csv
import json
from pathlib import Path

summary_json = Path("${SUMMARY_JSON}")
summary_csv = Path("${SUMMARY_CSV}")
eval_paths = sorted(Path("outputs/eval/capacity_adapters").glob("*/*_adapter_eval.json"))
rows = []
for path in eval_paths:
    data = json.loads(path.read_text())
    rows.append(
        {
            "path": str(path),
            "adapter_checkpoint_path": data.get("adapter_checkpoint_path"),
            "rmse_px": data.get("adapted_linear_probe_rmse_pixels"),
            "probe_vs_mean_ratio": data.get("adapted_linear_probe_vs_mean_rmse_ratio"),
            "rollout_ratio": data.get("adapted_rollout_vs_persistence_mse_ratio"),
            "pair_after": data.get("paired_latent_mse_after_adapter"),
            "delta_l2": data.get("delta_l2"),
            "scale_mean": data.get("scale_mean"),
            "residual_param_l2": data.get("residual_param_l2"),
            "adapter_trainable_parameters": data.get("adapter_trainable_parameters"),
        }
    )

rows.sort(
    key=lambda row: (
        float("inf") if row["rmse_px"] is None else row["rmse_px"],
        float("inf") if row["rollout_ratio"] is None else row["rollout_ratio"],
    )
)
summary = {"rows": rows}
summary_json.parent.mkdir(parents=True, exist_ok=True)
summary_csv.parent.mkdir(parents=True, exist_ok=True)
summary_json.write_text(json.dumps(summary, indent=2) + "\n")

if rows:
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

print(json.dumps(summary, indent=2))
PY

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Capacity adapter sweep DONE"
