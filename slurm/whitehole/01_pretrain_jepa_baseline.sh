#!/bin/bash
# First real JEPA source baseline for the two-room environment.
#
# This is the baseline to freeze before testing latent shift/adaptation methods:
# 3M reward-free transitions, original-style len=17 / n_steps=16 JEPA L1
# pretraining, no planning/probing eval during pretraining, checkpoint every
# epoch so a 4h Slurm allocation can be resumed.
#
# Submit from /valhalla/projects/bg-eng-01/WhiteHole:
#   sbatch slurm/whitehole/01_pretrain_jepa_baseline.sh
#
# Optional knobs:
#   NUM_TRANSITIONS=100000 sbatch slurm/whitehole/01_pretrain_jepa_baseline.sh
#   EPOCHS=3 OUTPUT_DIR=two_rooms_jepa_probe_100k sbatch ...
#   AUTO_CHAIN=0 sbatch ...   # run only one 4h job, without a successor

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=wh_jepa_base
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH -o /valhalla/projects/bg-eng-01/WhiteHole/logs/wh_jepa_base.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/WhiteHole/logs/wh_jepa_base.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/WhiteHole"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"

DATA_CONFIG="${DATA_CONFIG:-configs/two_rooms_baseline_data.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/two_rooms_baseline_jepa.yaml}"
DATA_PATH="${DATA_PATH:-outputs/data/two_rooms_len17_3m.npz}"
NUM_TRANSITIONS="${NUM_TRANSITIONS:-3000000}"
EPOCHS="${EPOCHS:-10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/pldm}"
OUTPUT_DIR="${OUTPUT_DIR:-two_rooms_jepa_baseline_len17_3m}"
# OfflineWallDataset moves samples to CUDA in __getitem__, so multiprocessing
# workers hit "Cannot re-initialize CUDA in forked subprocess". Keep this at 0
# unless the dataset/device path is refactored.
NUM_WORKERS="${NUM_WORKERS:-0}"
INSTALL_DEPS="${INSTALL_DEPS:-true}"
AUTO_CHAIN="${AUTO_CHAIN:-1}"
SELF_SCRIPT="slurm/whitehole/01_pretrain_jepa_baseline.sh"

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
mkdir -p logs outputs/data "${OUTPUT_ROOT}/${OUTPUT_DIR}"

echo ""
echo "===================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] WhiteHole JEPA baseline"
echo "  data_config=${DATA_CONFIG}"
echo "  train_config=${TRAIN_CONFIG}"
echo "  data_path=${DATA_PATH}"
echo "  transitions=${NUM_TRANSITIONS}"
echo "  epochs=${EPOCHS}"
echo "  output=${OUTPUT_ROOT}/${OUTPUT_DIR}"
echo "===================================================="
echo ""

check_python_deps() {
python - <<'PY'
missing = []
for name in (
    "torch",
    "omegaconf",
    "gymnasium",
    "wandb",
    "scipy",
    "tqdm",
    "matplotlib",
    "arm_pytorch_utilities",
):
    try:
        __import__(name)
    except Exception as exc:
        missing.append(f"{name}: {type(exc).__name__}: {exc}")

if missing:
    print("Missing Python dependencies in the selected environment:")
    for item in missing:
        print(f"  {item}")
    raise SystemExit(1)
PY
}

if ! check_python_deps; then
    if [ "${INSTALL_DEPS}" != "true" ]; then
        echo "INSTALL_DEPS=${INSTALL_DEPS}; refusing to install missing dependencies."
        exit 1
    fi

    echo ""
    echo "Installing missing WhiteHole baseline dependencies into ${VIRTUAL_ENV}..."
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

    echo ""
    echo "Rechecking Python dependencies..."
    check_python_deps
fi

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required because pldm/train.py currently uses .cuda() directly.")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i} = {props.name} | mem={props.total_memory/1e9:.1f} GB")
PY

if [ ! -f "${DATA_PATH}" ]; then
    echo "Dataset not found; generating ${NUM_TRANSITIONS} transitions..."
    python pldm_envs/wall/generate_data.py \
        --config_path "${DATA_CONFIG}" \
        --num_transitions "${NUM_TRANSITIONS}" \
        --output_path "${DATA_PATH}"
else
    echo "Dataset exists; reusing ${DATA_PATH}"
fi

source slurm/whitehole/_autochain_epochs.sh

T0=$(date +%s)

python -m pldm.train \
    --configs "${TRAIN_CONFIG}" \
    --values \
        epochs="${EPOCHS}" \
        output_root="${OUTPUT_ROOT}" \
        output_dir="${OUTPUT_DIR}" \
        data.offline_wall_config.offline_data_path="${DATA_PATH}" \
        data.num_workers="${NUM_WORKERS}" \
        resume_if_possible=true \
        wandb=false \
        train_only=true

T1=$(date +%s)
ELAPSED=$((T1 - T0))

echo ""
echo "===================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] WhiteHole JEPA baseline DONE"
echo "  elapsed: ${ELAPSED} s = $((ELAPSED / 60)) min"
echo "  output=${OUTPUT_ROOT}/${OUTPUT_DIR}"
echo "===================================================="
