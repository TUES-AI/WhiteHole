#!/bin/bash
# Train the first appearance adapter: one learned latent delta vector.
#
# Submit from /valhalla/projects/bg-eng-01/WhiteHole:
#   sbatch slurm/whitehole/03_train_appearance_adapter.sh
#
# Quick test:
#   EPOCHS=1 MAX_TRAIN_BATCHES_PER_EPOCH=50 VAL_BATCHES=8 sbatch ...
#
# AUTO_EVAL=1 submits slurm/whitehole/04_eval_appearance_adapter.sh after a
# successful train, using ${OUTPUT_DIR}/adapter_latest.ckpt.

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=wh_adapter
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o /valhalla/projects/bg-eng-01/WhiteHole/logs/wh_adapter.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/WhiteHole/logs/wh_adapter.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/WhiteHole"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
CONFIG="${CONFIG:-configs/adaptation/two_rooms_medium_delta_proposal.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/whitehole/two_rooms_jepa_baseline_len17_3m/epoch=10_sample_step=2072576.ckpt}"
DATA_PATH="${DATA_PATH:-outputs/data/two_rooms_len17_3m.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/adaptation/two_rooms_medium_delta_proposal_3ep}"
OUTPUT_DIR="${OUTPUT_DIR%/}"
APPEARANCE_SHIFT="${APPEARANCE_SHIFT:-medium}"
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
PAIR_ALIGNMENT_WEIGHT="${PAIR_ALIGNMENT_WEIGHT:-0.0}"
SOURCE_IDENTITY_WEIGHT="${SOURCE_IDENTITY_WEIGHT:-0.0}"
VARIANCE_ALIGNMENT_WEIGHT="${VARIANCE_ALIGNMENT_WEIGHT:-0.0}"
COVARIANCE_ALIGNMENT_WEIGHT="${COVARIANCE_ALIGNMENT_WEIGHT:-0.0}"
COVARIANCE_SAMPLES="${COVARIANCE_SAMPLES:-512}"
GRADIENT_CLIP_NORM="${GRADIENT_CLIP_NORM:-1.0}"
INSTALL_DEPS="${INSTALL_DEPS:-true}"
AUTO_EVAL="${AUTO_EVAL:-1}"
EVAL_SCRIPT="${EVAL_SCRIPT:-slurm/whitehole/04_eval_appearance_adapter.sh}"
RUN_NAME="${OUTPUT_DIR##*/}"
EVAL_ADAPTER_CHECKPOINT="${EVAL_ADAPTER_CHECKPOINT:-${OUTPUT_DIR}/adapter_latest.ckpt}"
EVAL_OUTPUT_JSON="${EVAL_OUTPUT_JSON:-outputs/eval/${RUN_NAME}_adapter_eval.json}"

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
mkdir -p logs "${OUTPUT_DIR}"

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
[ -f "${DATA_PATH}" ] || { echo "Missing data file: ${DATA_PATH}"; exit 1; }

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for this Discoverer adapter job.")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i} = {props.name} | mem={props.total_memory/1e9:.1f} GB")
PY

echo ""
echo "===================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] WhiteHole appearance adapter train"
echo "  config=${CONFIG}"
echo "  checkpoint=${CHECKPOINT}"
echo "  data=${DATA_PATH}"
echo "  output_dir=${OUTPUT_DIR}"
echo "  appearance_shift=${APPEARANCE_SHIFT}"
echo "  epochs=${EPOCHS}"
echo "  max_train_batches_per_epoch=${MAX_TRAIN_BATCHES_PER_EPOCH}"
echo "===================================================="
echo ""

T0=$(date +%s)

python -m whitehole.adaptation.train \
    --configs "${CONFIG}" \
    --values \
        source_checkpoint_path="${CHECKPOINT}" \
        output_dir="${OUTPUT_DIR}" \
        epochs="${EPOCHS}" \
        lr="${LR}" \
        weight_decay="${WEIGHT_DECAY}" \
        max_train_batches_per_epoch="${MAX_TRAIN_BATCHES_PER_EPOCH}" \
        delta_init_batches="${DELTA_INIT_BATCHES}" \
        source_scale_batches="${SOURCE_SCALE_BATCHES}" \
        val_batches="${VAL_BATCHES}" \
        gradient_clip_norm="${GRADIENT_CLIP_NORM}" \
        data.source_data_path="${DATA_PATH}" \
        data.appearance_shift="${APPEARANCE_SHIFT}" \
        data.batch_size="${BATCH_SIZE}" \
        data.num_workers="${NUM_WORKERS}" \
        objectives.horizon="${HORIZON}" \
        objectives.alignment_weight="${ALIGNMENT_WEIGHT}" \
        objectives.multistep_weight="${MULTISTEP_WEIGHT}" \
        objectives.multistep_discount="${MULTISTEP_DISCOUNT}" \
        objectives.local_isometry_weight="${LOCAL_ISOMETRY_WEIGHT}" \
        objectives.local_isometry_samples="${LOCAL_ISOMETRY_SAMPLES}" \
        objectives.identity_prior_weight="${IDENTITY_PRIOR_WEIGHT}" \
        objectives.pair_alignment_weight="${PAIR_ALIGNMENT_WEIGHT}" \
        objectives.source_identity_weight="${SOURCE_IDENTITY_WEIGHT}" \
        objectives.variance_alignment_weight="${VARIANCE_ALIGNMENT_WEIGHT}" \
        objectives.covariance_alignment_weight="${COVARIANCE_ALIGNMENT_WEIGHT}" \
        objectives.covariance_samples="${COVARIANCE_SAMPLES}"

T1=$(date +%s)
ELAPSED=$((T1 - T0))

echo ""
echo "===================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] WhiteHole appearance adapter train DONE"
echo "  elapsed: ${ELAPSED} s = $((ELAPSED / 60)) min"
echo "  output_dir=${OUTPUT_DIR}"
echo "  auto_eval=${AUTO_EVAL}"
echo "  eval_output_json=${EVAL_OUTPUT_JSON}"
echo "===================================================="

if [ "${AUTO_EVAL}" = "1" ] || [ "${AUTO_EVAL}" = "true" ]; then
    [ -f "${EVAL_ADAPTER_CHECKPOINT}" ] || {
        echo "Missing adapter checkpoint for eval: ${EVAL_ADAPTER_CHECKPOINT}"
        exit 1
    }

    echo ""
    echo "Submitting adapter eval job..."
    CONFIG="${CONFIG}" \
    CHECKPOINT="${CHECKPOINT}" \
    ADAPTER_CHECKPOINT="${EVAL_ADAPTER_CHECKPOINT}" \
    DATA_PATH="${DATA_PATH}" \
    APPEARANCE_SHIFT="${APPEARANCE_SHIFT}" \
    OUTPUT_JSON="${EVAL_OUTPUT_JSON}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    sbatch "${EVAL_SCRIPT}"
else
    echo ""
    echo "AUTO_EVAL=${AUTO_EVAL}; skipping adapter eval submission."
fi
