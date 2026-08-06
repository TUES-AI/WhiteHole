#!/bin/bash
# Diagnose where latent appearance adapters lose source-compatible semantics.
#
# Submit from /valhalla/projects/bg-eng-01/WhiteHole:
#   sbatch slurm/whitehole/11_diagnose_adapter_collapse.sh

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=wh_collapse
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o /valhalla/projects/bg-eng-01/WhiteHole/logs/wh_collapse.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/WhiteHole/logs/wh_collapse.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/WhiteHole"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
OUTPUT_JSON="${OUTPUT_JSON:-outputs/eval/adapter_collapse/diagnosis.json}"
OUTPUT_CSV="${OUTPUT_CSV:-outputs/eval/adapter_collapse/diagnosis.csv}"

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

python scripts/diagnose_adapter_collapse.py \
    --adapter "delta_pairw_1d0:outputs/adaptation/delta_anchor_init64/two_rooms_medium_delta_pairw_1d0_3ep/adapter_latest.ckpt" \
    --adapter "diagonal_pairw_0d3:outputs/adaptation/diagonal_affine_init64/two_rooms_medium_diagaff_pairw_0d3_3ep/adapter_latest.ckpt" \
    --adapter "lowrank_pairw_0d3:outputs/adaptation/capacity_adapters/lowrank_r32/two_rooms_medium_lowrank_pairw_0d3_3ep/adapter_latest.ckpt" \
    --adapter "mlp_pairw_0d3:outputs/adaptation/capacity_adapters/mlp_h256/two_rooms_medium_mlp_pairw_0d3_3ep/adapter_latest.ckpt" \
    --output-json "${OUTPUT_JSON}" \
    --output-csv "${OUTPUT_CSV}"

echo ""
echo "Wrote ${OUTPUT_JSON}"
echo "Wrote ${OUTPUT_CSV}"
