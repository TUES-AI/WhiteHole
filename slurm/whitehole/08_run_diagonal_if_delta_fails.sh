#!/bin/bash
# Conditionally run diagonal-affine adaptation if strong delta anchors fail.
#
# Submit after the strong delta sweep:
#   sbatch --dependency=afterok:<delta_job_id> \
#     slurm/whitehole/08_run_diagonal_if_delta_fails.sh

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=wh_diag_if
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o /valhalla/projects/bg-eng-01/WhiteHole/logs/wh_diag_if.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/WhiteHole/logs/wh_diag_if.%j.err

set -euo pipefail

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/WhiteHole"
DELTA_EVAL_ROOT="${DELTA_EVAL_ROOT:-outputs/eval/delta_anchor_init64}"
DELTA_RUN_PREFIX="${DELTA_RUN_PREFIX:-two_rooms_medium_delta_pairw}"
DELTA_WEIGHT_TAGS="${DELTA_WEIGHT_TAGS:-1d0 3d0 10d0}"
MAX_ACCEPTABLE_RMSE="${MAX_ACCEPTABLE_RMSE:-15.0}"
MAX_ACCEPTABLE_ROLLOUT_RATIO="${MAX_ACCEPTABLE_ROLLOUT_RATIO:-1000.0}"

DIAG_CONFIG="${DIAG_CONFIG:-configs/adaptation/two_rooms_medium_diagonal_affine.yaml}"
DIAG_PAIR_ALIGNMENT_WEIGHTS="${DIAG_PAIR_ALIGNMENT_WEIGHTS:-0.3 1.0 3.0}"
DIAG_OUTPUT_ROOT="${DIAG_OUTPUT_ROOT:-outputs/adaptation/diagonal_affine_init64}"
DIAG_EVAL_OUTPUT_ROOT="${DIAG_EVAL_OUTPUT_ROOT:-outputs/eval/diagonal_affine_init64}"
DIAG_RUN_PREFIX="${DIAG_RUN_PREFIX:-two_rooms_medium_diagaff_pairw}"
DIAG_DELTA_INIT_BATCHES="${DIAG_DELTA_INIT_BATCHES:-64}"
DIAG_LOCAL_ISOMETRY_WEIGHT="${DIAG_LOCAL_ISOMETRY_WEIGHT:-0.1}"
SWEEP_SCRIPT="${SWEEP_SCRIPT:-slurm/whitehole/07_run_delta_pair_anchor_sweep.sh}"

[ -d "${PROJECT_DIR}" ] || { echo "Missing project dir: ${PROJECT_DIR}"; exit 1; }

cd "${PROJECT_DIR}"
mkdir -p logs "${DIAG_OUTPUT_ROOT}" "${DIAG_EVAL_OUTPUT_ROOT}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking strong delta-anchor results"
echo "  eval_root=${DELTA_EVAL_ROOT}"
echo "  max_rmse=${MAX_ACCEPTABLE_RMSE}"
echo "  max_rollout_ratio=${MAX_ACCEPTABLE_ROLLOUT_RATIO}"

if python - <<PY
import json
import os
import sys

eval_root = os.environ.get("DELTA_EVAL_ROOT", "${DELTA_EVAL_ROOT}")
prefix = os.environ.get("DELTA_RUN_PREFIX", "${DELTA_RUN_PREFIX}")
tags = os.environ.get("DELTA_WEIGHT_TAGS", "${DELTA_WEIGHT_TAGS}").split()
max_rmse = float(os.environ.get("MAX_ACCEPTABLE_RMSE", "${MAX_ACCEPTABLE_RMSE}"))
max_rollout = float(
    os.environ.get("MAX_ACCEPTABLE_ROLLOUT_RATIO", "${MAX_ACCEPTABLE_ROLLOUT_RATIO}")
)

rows = []
passing = []
for tag in tags:
    path = os.path.join(eval_root, f"{prefix}_{tag}_3ep_adapter_eval.json")
    if not os.path.exists(path):
        print(f"missing_delta_eval={path}")
        continue

    with open(path) as f:
        data = json.load(f)
    rmse = float(data["adapted_linear_probe_rmse_pixels"])
    rollout = float(data["adapted_rollout_vs_persistence_mse_ratio"])
    pair_after = float(data["paired_latent_mse_after_adapter"])
    print(
        f"delta_tag={tag} adapted_rmse={rmse:.4f} "
        f"rollout_ratio={rollout:.4f} pair_after={pair_after:.4f}"
    )
    row = {
        "tag": tag,
        "rmse": rmse,
        "rollout": rollout,
        "pair_after": pair_after,
    }
    rows.append(row)
    if rmse <= max_rmse and rollout < max_rollout:
        passing.append(row)

if not rows:
    print("No delta eval files found; running diagonal-affine follow-up.")
    raise SystemExit(2)

best = min(rows, key=lambda row: (row["rmse"], row["rollout"]))
print(
    f"best_delta_tag={best['tag']} best_rmse={best['rmse']:.4f} "
    f"best_rollout_ratio={best['rollout']:.4f} "
    f"best_pair_after={best['pair_after']:.4f}"
)

if passing:
    best_passing = min(passing, key=lambda row: (row["rmse"], row["rollout"]))
    print(
        f"passing_delta_tag={best_passing['tag']} "
        f"passing_rmse={best_passing['rmse']:.4f} "
        f"passing_rollout_ratio={best_passing['rollout']:.4f} "
        f"passing_pair_after={best_passing['pair_after']:.4f}"
    )
    print("Strong delta-anchor sweep met the target; skipping diagonal-affine.")
    raise SystemExit(0)

print("Strong delta-anchor sweep missed the target; running diagonal-affine.")
raise SystemExit(2)
PY
then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Conditional follow-up DONE"
    exit 0
fi

echo ""
echo "===================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting diagonal-affine follow-up"
echo "  config=${DIAG_CONFIG}"
echo "  weights=${DIAG_PAIR_ALIGNMENT_WEIGHTS}"
echo "  local_isometry_weight=${DIAG_LOCAL_ISOMETRY_WEIGHT}"
echo "===================================================="

CONFIG="${DIAG_CONFIG}" \
PAIR_ALIGNMENT_WEIGHTS="${DIAG_PAIR_ALIGNMENT_WEIGHTS}" \
DELTA_INIT_BATCHES="${DIAG_DELTA_INIT_BATCHES}" \
OUTPUT_ROOT="${DIAG_OUTPUT_ROOT}" \
EVAL_OUTPUT_ROOT="${DIAG_EVAL_OUTPUT_ROOT}" \
RUN_PREFIX="${DIAG_RUN_PREFIX}" \
LOCAL_ISOMETRY_WEIGHT="${DIAG_LOCAL_ISOMETRY_WEIGHT}" \
bash "${SWEEP_SCRIPT}"

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Conditional diagonal-affine follow-up DONE"
