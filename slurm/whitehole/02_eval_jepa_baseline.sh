#!/bin/bash
# Diagnostic evaluation for the pretrained two-room JEPA source baseline.
#
# Submit from /valhalla/projects/bg-eng-01/WhiteHole:
#   sbatch slurm/whitehole/02_eval_jepa_baseline.sh

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=wh_jepa_eval
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o /valhalla/projects/bg-eng-01/WhiteHole/logs/wh_jepa_eval.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/WhiteHole/logs/wh_jepa_eval.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/WhiteHole"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"

CONFIG="${CONFIG:-configs/two_rooms_baseline_jepa.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/whitehole/two_rooms_jepa_baseline_len17_3m/epoch=10_sample_step=2072576.ckpt}"
EVAL_APPEARANCE_SHIFT="${EVAL_APPEARANCE_SHIFT:-source}"
OUTPUT_JSON="${OUTPUT_JSON:-outputs/eval/two_rooms_jepa_baseline_${EVAL_APPEARANCE_SHIFT}_eval.json}"

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
mkdir -p logs "$(dirname "${OUTPUT_JSON}")"

python -m pip install \
    omegaconf \
    gymnasium \
    wandb \
    scipy \
    tqdm \
    matplotlib \
    pyyaml \
    imageio \
    pillow \
    arm-pytorch-utilities

[ -f "${CHECKPOINT}" ] || { echo "Missing checkpoint: ${CHECKPOINT}"; exit 1; }

echo ""
echo "===================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] WhiteHole JEPA eval"
echo "  config=${CONFIG}"
echo "  checkpoint=${CHECKPOINT}"
echo "  output_json=${OUTPUT_JSON}"
echo "  eval_appearance_shift=${EVAL_APPEARANCE_SHIFT}"
echo "===================================================="
echo ""

python scripts/eval_jepa_baseline.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --output-json "${OUTPUT_JSON}" \
    --eval-appearance-shift "${EVAL_APPEARANCE_SHIFT}"

echo ""
echo "Wrote ${OUTPUT_JSON}"
