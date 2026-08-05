#!/bin/bash
# Evaluate a trained appearance adapter against the shifted two-room environment.
#
# Submit from /valhalla/projects/bg-eng-01/WhiteHole:
#   sbatch slurm/whitehole/04_eval_appearance_adapter.sh

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=wh_adapt_eval
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o /valhalla/projects/bg-eng-01/WhiteHole/logs/wh_adapt_eval.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/WhiteHole/logs/wh_adapt_eval.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/WhiteHole"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
CONFIG="${CONFIG:-configs/adaptation/two_rooms_medium_delta_proposal.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/pldm/two_rooms_jepa_baseline_len17_3m/epoch=10_sample_step=2072576.ckpt}"
ADAPTER_CHECKPOINT="${ADAPTER_CHECKPOINT:-outputs/adaptation/two_rooms_medium_delta_proposal_3ep/adapter_latest.ckpt}"
DATA_PATH="${DATA_PATH:-outputs/data/two_rooms_len17_3m.npz}"
APPEARANCE_SHIFT="${APPEARANCE_SHIFT:-medium}"
OUTPUT_JSON="${OUTPUT_JSON:-outputs/eval/two_rooms_medium_delta_proposal_3ep_adapter_eval.json}"
PROBE_TRAIN_BATCHES="${PROBE_TRAIN_BATCHES:-48}"
PROBE_VAL_BATCHES="${PROBE_VAL_BATCHES:-16}"
PROBE_STEPS="${PROBE_STEPS:-400}"
ROLLOUT_BATCHES="${ROLLOUT_BATCHES:-32}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"
INSTALL_DEPS="${INSTALL_DEPS:-true}"

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

    check_python_deps
fi

[ -f "${CHECKPOINT}" ] || { echo "Missing checkpoint: ${CHECKPOINT}"; exit 1; }
[ -f "${ADAPTER_CHECKPOINT}" ] || { echo "Missing adapter checkpoint: ${ADAPTER_CHECKPOINT}"; exit 1; }
[ -f "${DATA_PATH}" ] || { echo "Missing data file: ${DATA_PATH}"; exit 1; }

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for this Discoverer adapter eval job.")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i} = {props.name} | mem={props.total_memory/1e9:.1f} GB")
PY

echo ""
echo "===================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] WhiteHole appearance adapter eval"
echo "  config=${CONFIG}"
echo "  checkpoint=${CHECKPOINT}"
echo "  adapter_checkpoint=${ADAPTER_CHECKPOINT}"
echo "  data=${DATA_PATH}"
echo "  output_json=${OUTPUT_JSON}"
echo "  appearance_shift=${APPEARANCE_SHIFT}"
echo "===================================================="
echo ""

python -m pldm.adaptation.eval \
    --configs "${CONFIG}" \
    --values \
        source_checkpoint_path="${CHECKPOINT}" \
        adapter_checkpoint_path="${ADAPTER_CHECKPOINT}" \
        output_json="${OUTPUT_JSON}" \
        data.source_data_path="${DATA_PATH}" \
        data.appearance_shift="${APPEARANCE_SHIFT}" \
        data.batch_size="${BATCH_SIZE}" \
        data.num_workers="${NUM_WORKERS}" \
        probe_train_batches="${PROBE_TRAIN_BATCHES}" \
        probe_val_batches="${PROBE_VAL_BATCHES}" \
        probe_steps="${PROBE_STEPS}" \
        rollout_batches="${ROLLOUT_BATCHES}"

echo ""
echo "Wrote ${OUTPUT_JSON}"
