#!/bin/bash
# Train a tiny input-channel FiLM/affine adapter before the frozen JEPA encoder.
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

OUTPUT_DIR="${OUTPUT_DIR:-outputs/adaptation/input_film/two_rooms_medium_input_affine_3ep}"
EVAL_JSON="${EVAL_JSON:-outputs/eval/input_film/two_rooms_medium_input_affine_3ep_eval.json}"
EVAL_CSV="${EVAL_CSV:-outputs/eval/input_film/two_rooms_medium_input_affine_3ep_eval.csv}"
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
echo "  output_dir=${OUTPUT_DIR}"
echo "  eval_json=${EVAL_JSON}"
echo "  epochs=${EPOCHS}"
echo "  max_train_batches_per_epoch=${MAX_TRAIN_BATCHES_PER_EPOCH}"
echo "  lr=${LR}"
echo "===================================================="

python scripts/train_input_film_adapter.py \
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
    --image-pair-weight "${IMAGE_PAIR_WEIGHT}"

echo ""
echo "Wrote ${EVAL_JSON}"
echo "Wrote ${EVAL_CSV}"
