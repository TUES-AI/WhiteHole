#!/bin/bash
# Run a pair-anchor adapter sweep inside one Slurm allocation.
#
# Submit from /valhalla/projects/bg-eng-01/WhiteHole:
#   sbatch slurm/whitehole/07_run_delta_pair_anchor_sweep.sh

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=wh_pair_sweep
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o /valhalla/projects/bg-eng-01/WhiteHole/logs/wh_pair_sweep.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/WhiteHole/logs/wh_pair_sweep.%j.err

set -euo pipefail

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/WhiteHole"
CONFIG="${CONFIG:-configs/adaptation/two_rooms_medium_delta_proposal.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/whitehole/two_rooms_jepa_baseline_len17_3m/epoch=10_sample_step=2072576.ckpt}"
DATA_PATH="${DATA_PATH:-outputs/data/two_rooms_len17_3m.npz}"
APPEARANCE_SHIFT="${APPEARANCE_SHIFT:-medium}"
PAIR_ALIGNMENT_WEIGHTS="${PAIR_ALIGNMENT_WEIGHTS:-0.01 0.03 0.1 0.3}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
MAX_TRAIN_BATCHES_PER_EPOCH="${MAX_TRAIN_BATCHES_PER_EPOCH:-1500}"
DELTA_INIT_BATCHES="${DELTA_INIT_BATCHES:-0}"
SOURCE_SCALE_BATCHES="${SOURCE_SCALE_BATCHES:-64}"
VAL_BATCHES="${VAL_BATCHES:-64}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"
HORIZON="${HORIZON:-15}"
ALIGNMENT_WEIGHT="${ALIGNMENT_WEIGHT:-1.0}"
MULTISTEP_WEIGHT="${MULTISTEP_WEIGHT:-1.0}"
MULTISTEP_DISCOUNT="${MULTISTEP_DISCOUNT:-1.0}"
LOCAL_ISOMETRY_WEIGHT="${LOCAL_ISOMETRY_WEIGHT:-1.0}"
LOCAL_ISOMETRY_SAMPLES="${LOCAL_ISOMETRY_SAMPLES:-256}"
IDENTITY_PRIOR_WEIGHT="${IDENTITY_PRIOR_WEIGHT:-0.0001}"
SOURCE_IDENTITY_WEIGHT="${SOURCE_IDENTITY_WEIGHT:-0.0}"
VARIANCE_ALIGNMENT_WEIGHT="${VARIANCE_ALIGNMENT_WEIGHT:-0.0}"
COVARIANCE_ALIGNMENT_WEIGHT="${COVARIANCE_ALIGNMENT_WEIGHT:-0.0}"
COVARIANCE_SAMPLES="${COVARIANCE_SAMPLES:-512}"
GRADIENT_CLIP_NORM="${GRADIENT_CLIP_NORM:-1.0}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-slurm/whitehole/03_train_appearance_adapter.sh}"
EVAL_SCRIPT="${EVAL_SCRIPT:-slurm/whitehole/04_eval_appearance_adapter.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/adaptation}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-outputs/eval}"
RUN_PREFIX="${RUN_PREFIX:-two_rooms_medium_delta_pairw}"
RUN_SUFFIX="${RUN_SUFFIX:-3ep}"

[ -d "${PROJECT_DIR}" ] || { echo "Missing project dir: ${PROJECT_DIR}"; exit 1; }

cd "${PROJECT_DIR}"
mkdir -p logs "${OUTPUT_ROOT}" "${EVAL_OUTPUT_ROOT}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting pair-anchor adapter sweep"
echo "  weights=${PAIR_ALIGNMENT_WEIGHTS}"
echo "  epochs=${EPOCHS}"
echo "  run_prefix=${RUN_PREFIX}"
echo "  run_suffix=${RUN_SUFFIX}"
echo "  gpu=${CUDA_VISIBLE_DEVICES:-unset}"

for weight in ${PAIR_ALIGNMENT_WEIGHTS}; do
    tag="${weight//./d}"
    tag="${tag//-/m}"
    run_name="${RUN_PREFIX}_${tag}_${RUN_SUFFIX}"
    output_dir="${OUTPUT_ROOT}/${run_name}"
    output_json="${EVAL_OUTPUT_ROOT}/${run_name}_adapter_eval.json"

    echo ""
    echo "===================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] pair_alignment_weight=${weight}"
    echo "  output_dir=${output_dir}"
    echo "  output_json=${output_json}"
    echo "===================================================="

    CONFIG="${CONFIG}" \
    CHECKPOINT="${CHECKPOINT}" \
    DATA_PATH="${DATA_PATH}" \
    OUTPUT_DIR="${output_dir}" \
    APPEARANCE_SHIFT="${APPEARANCE_SHIFT}" \
    EPOCHS="${EPOCHS}" \
    LR="${LR}" \
    WEIGHT_DECAY="${WEIGHT_DECAY}" \
    MAX_TRAIN_BATCHES_PER_EPOCH="${MAX_TRAIN_BATCHES_PER_EPOCH}" \
    DELTA_INIT_BATCHES="${DELTA_INIT_BATCHES}" \
    SOURCE_SCALE_BATCHES="${SOURCE_SCALE_BATCHES}" \
    VAL_BATCHES="${VAL_BATCHES}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    HORIZON="${HORIZON}" \
    ALIGNMENT_WEIGHT="${ALIGNMENT_WEIGHT}" \
    MULTISTEP_WEIGHT="${MULTISTEP_WEIGHT}" \
    MULTISTEP_DISCOUNT="${MULTISTEP_DISCOUNT}" \
    LOCAL_ISOMETRY_WEIGHT="${LOCAL_ISOMETRY_WEIGHT}" \
    LOCAL_ISOMETRY_SAMPLES="${LOCAL_ISOMETRY_SAMPLES}" \
    IDENTITY_PRIOR_WEIGHT="${IDENTITY_PRIOR_WEIGHT}" \
    PAIR_ALIGNMENT_WEIGHT="${weight}" \
    SOURCE_IDENTITY_WEIGHT="${SOURCE_IDENTITY_WEIGHT}" \
    VARIANCE_ALIGNMENT_WEIGHT="${VARIANCE_ALIGNMENT_WEIGHT}" \
    COVARIANCE_ALIGNMENT_WEIGHT="${COVARIANCE_ALIGNMENT_WEIGHT}" \
    COVARIANCE_SAMPLES="${COVARIANCE_SAMPLES}" \
    GRADIENT_CLIP_NORM="${GRADIENT_CLIP_NORM}" \
    AUTO_EVAL=0 \
    bash "${TRAIN_SCRIPT}"

    CONFIG="${CONFIG}" \
    CHECKPOINT="${CHECKPOINT}" \
    ADAPTER_CHECKPOINT="${output_dir}/adapter_latest.ckpt" \
    DATA_PATH="${DATA_PATH}" \
    APPEARANCE_SHIFT="${APPEARANCE_SHIFT}" \
    OUTPUT_JSON="${output_json}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    bash "${EVAL_SCRIPT}"
done

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pair-anchor adapter sweep DONE"
