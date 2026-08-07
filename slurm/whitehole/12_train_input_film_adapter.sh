#!/bin/bash
# Train a small input adapter before the frozen JEPA encoder.
#
# Submit from /valhalla/projects/bg-eng-01/WhiteHole:
#   sbatch slurm/whitehole/12_train_input_film_adapter.sh

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=wh_input_film
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o /valhalla/projects/bg-eng-01/WhiteHole/logs/wh_input_film.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/WhiteHole/logs/wh_input_film.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/WhiteHole"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"

APPEARANCE_SHIFT="${APPEARANCE_SHIFT:-medium}"
ADAPTER_FAMILY="${ADAPTER_FAMILY:-affine}"
RUN_NAME="${RUN_NAME:-two_rooms_${APPEARANCE_SHIFT}_input_${ADAPTER_FAMILY}_3ep}"
DATA_PATH="${DATA_PATH:-outputs/data/two_rooms_len17_3m.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/adaptation/input_film/${RUN_NAME}}"
EVAL_JSON="${EVAL_JSON:-outputs/eval/input_film/${RUN_NAME}_eval.json}"
EVAL_CSV="${EVAL_CSV:-outputs/eval/input_film/${RUN_NAME}_eval.csv}"
EPOCHS="${EPOCHS:-3}"
MAX_TRAIN_BATCHES_PER_EPOCH="${MAX_TRAIN_BATCHES_PER_EPOCH:-1000}"
LR="${LR:-0.02}"
PAIR_WEIGHT="${PAIR_WEIGHT:-1.0}"
SOURCE_ROLLOUT_WEIGHT="${SOURCE_ROLLOUT_WEIGHT:-1.0}"
SELF_ROLLOUT_WEIGHT="${SELF_ROLLOUT_WEIGHT:-0.1}"
VARIANCE_WEIGHT="${VARIANCE_WEIGHT:-1.0}"
COVARIANCE_WEIGHT="${COVARIANCE_WEIGHT:-0.05}"
IDENTITY_WEIGHT="${IDENTITY_WEIGHT:-0.001}"
IMAGE_PAIR_WEIGHT="${IMAGE_PAIR_WEIGHT:-0.0}"
CONV_HIDDEN_CHANNELS="${CONV_HIDDEN_CHANNELS:-16}"
CONV_LAYERS="${CONV_LAYERS:-3}"
CONV_RESIDUAL_SCALE="${CONV_RESIDUAL_SCALE:-1.0}"
CONV_ZERO_INIT="${CONV_ZERO_INIT:-true}"

[ -d "${PROJECT_DIR}" ] || { echo "Missing project dir: ${PROJECT_DIR}"; exit 1; }
[ -d "${VIRTUAL_ENV}" ] || { echo "Missing venv: ${VIRTUAL_ENV}"; exit 1; }

export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "${PROJECT_DIR}"
mkdir -p logs "${OUTPUT_DIR}" "$(dirname "${EVAL_JSON}")"

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for this Discoverer job.")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i} = {props.name} | mem={props.total_memory/1e9:.1f} GB")
PY

echo ""
echo "===================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Input FiLM encoder-adaptation control"
echo "  appearance_shift=${APPEARANCE_SHIFT}"
echo "  adapter_family=${ADAPTER_FAMILY}"
echo "  data_path=${DATA_PATH}"
echo "  output_dir=${OUTPUT_DIR}"
echo "  eval_json=${EVAL_JSON}"
echo "  epochs=${EPOCHS}"
echo "  max_train_batches_per_epoch=${MAX_TRAIN_BATCHES_PER_EPOCH}"
echo "  lr=${LR}"
echo "===================================================="

python scripts/train_input_film_adapter.py \
    --appearance-shift "${APPEARANCE_SHIFT}" \
    --adapter-family "${ADAPTER_FAMILY}" \
    --conv-hidden-channels "${CONV_HIDDEN_CHANNELS}" \
    --conv-layers "${CONV_LAYERS}" \
    --conv-residual-scale "${CONV_RESIDUAL_SCALE}" \
    --data-path "${DATA_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --eval-json "${EVAL_JSON}" \
    --eval-csv "${EVAL_CSV}" \
    --epochs "${EPOCHS}" \
    --max-train-batches-per-epoch "${MAX_TRAIN_BATCHES_PER_EPOCH}" \
    --lr "${LR}" \
    --pair-weight "${PAIR_WEIGHT}" \
    --source-rollout-weight "${SOURCE_ROLLOUT_WEIGHT}" \
    --self-rollout-weight "${SELF_ROLLOUT_WEIGHT}" \
    --variance-weight "${VARIANCE_WEIGHT}" \
    --covariance-weight "${COVARIANCE_WEIGHT}" \
    --identity-weight "${IDENTITY_WEIGHT}" \
    --image-pair-weight "${IMAGE_PAIR_WEIGHT}" \
    "$([ "${CONV_ZERO_INIT}" = "true" ] && echo --conv-zero-init || echo --no-conv-zero-init)"

echo ""
echo "Wrote ${EVAL_JSON}"
echo "Wrote ${EVAL_CSV}"
