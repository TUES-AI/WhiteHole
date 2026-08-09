#!/bin/bash
# Compare source, shifted, delta-adapted, and diagonal-affine-adapted JEPA evals.
#
# Submit from /valhalla/projects/bg-eng-01/WhiteHole:
#   sbatch slurm/whitehole/09_eval_medium_shift_downstream.sh

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=wh_med_eval
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o /valhalla/projects/bg-eng-01/WhiteHole/logs/wh_med_eval.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/WhiteHole/logs/wh_med_eval.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/WhiteHole"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"

CONFIG="${CONFIG:-configs/two_rooms_baseline_jepa.yaml}"
DELTA_CONFIG="${DELTA_CONFIG:-configs/adaptation/two_rooms_medium_adapter.yaml}"
DIAGONAL_CONFIG="${DIAGONAL_CONFIG:-configs/adaptation/two_rooms_medium_diagonal_affine.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/whitehole/two_rooms_jepa_baseline_len17_3m/epoch=10_sample_step=2072576.ckpt}"
DATA_PATH="${DATA_PATH:-outputs/data/two_rooms_len17_3m.npz}"
DELTA_ADAPTER_CHECKPOINT="${DELTA_ADAPTER_CHECKPOINT:-outputs/adaptation/delta_anchor_init64/two_rooms_medium_delta_pairw_1d0_3ep/adapter_latest.ckpt}"
DIAGONAL_ADAPTER_CHECKPOINT="${DIAGONAL_ADAPTER_CHECKPOINT:-outputs/adaptation/diagonal_affine_init64/two_rooms_medium_diagaff_pairw_0d3_3ep/adapter_latest.ckpt}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/eval/medium_shift_downstream}"
SOURCE_JSON="${SOURCE_JSON:-${OUTPUT_ROOT}/source_baseline_eval.json}"
MEDIUM_JSON="${MEDIUM_JSON:-${OUTPUT_ROOT}/medium_unadapted_eval.json}"
DELTA_JSON="${DELTA_JSON:-${OUTPUT_ROOT}/medium_delta_pairw_1d0_eval.json}"
DIAGONAL_JSON="${DIAGONAL_JSON:-${OUTPUT_ROOT}/medium_diagonal_affine_pairw_0d3_eval.json}"
SUMMARY_JSON="${SUMMARY_JSON:-${OUTPUT_ROOT}/comparison_summary.json}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUTPUT_ROOT}/comparison_summary.csv}"

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
mkdir -p logs "${OUTPUT_ROOT}"

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
[ -f "${DELTA_ADAPTER_CHECKPOINT}" ] || { echo "Missing delta adapter: ${DELTA_ADAPTER_CHECKPOINT}"; exit 1; }
[ -f "${DIAGONAL_ADAPTER_CHECKPOINT}" ] || { echo "Missing diagonal adapter: ${DIAGONAL_ADAPTER_CHECKPOINT}"; exit 1; }

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for this Discoverer eval job.")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i} = {props.name} | mem={props.total_memory/1e9:.1f} GB")
PY

echo ""
echo "===================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Medium-shift downstream comparison"
echo "  checkpoint=${CHECKPOINT}"
echo "  data=${DATA_PATH}"
echo "  delta_adapter=${DELTA_ADAPTER_CHECKPOINT}"
echo "  diagonal_adapter=${DIAGONAL_ADAPTER_CHECKPOINT}"
echo "  output_root=${OUTPUT_ROOT}"
echo "===================================================="

echo ""
echo "[1/5] Source baseline eval"
python scripts/eval_jepa_baseline.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --output-json "${SOURCE_JSON}" \
    --eval-appearance-shift source \
    --batch-size "${BATCH_SIZE}" \
    --probe-train-batches "${PROBE_TRAIN_BATCHES}" \
    --probe-val-batches "${PROBE_VAL_BATCHES}" \
    --probe-steps "${PROBE_STEPS}" \
    --rollout-batches "${ROLLOUT_BATCHES}"

echo ""
echo "[2/5] Medium-shift unadapted eval"
python scripts/eval_jepa_baseline.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --output-json "${MEDIUM_JSON}" \
    --eval-appearance-shift medium \
    --batch-size "${BATCH_SIZE}" \
    --probe-train-batches "${PROBE_TRAIN_BATCHES}" \
    --probe-val-batches "${PROBE_VAL_BATCHES}" \
    --probe-steps "${PROBE_STEPS}" \
    --rollout-batches "${ROLLOUT_BATCHES}"

echo ""
echo "[3/5] Medium-shift delta adapter eval"
python -m whitehole.adaptation.eval \
    --configs "${DELTA_CONFIG}" \
    --values \
        source_checkpoint_path="${CHECKPOINT}" \
        adapter_checkpoint_path="${DELTA_ADAPTER_CHECKPOINT}" \
        output_json="${DELTA_JSON}" \
        data.source_data_path="${DATA_PATH}" \
        data.appearance_shift=medium \
        data.batch_size="${BATCH_SIZE}" \
        data.num_workers="${NUM_WORKERS}" \
        probe_train_batches="${PROBE_TRAIN_BATCHES}" \
        probe_val_batches="${PROBE_VAL_BATCHES}" \
        probe_steps="${PROBE_STEPS}" \
        rollout_batches="${ROLLOUT_BATCHES}"

echo ""
echo "[4/5] Medium-shift diagonal-affine adapter eval"
python -m whitehole.adaptation.eval \
    --configs "${DIAGONAL_CONFIG}" \
    --values \
        source_checkpoint_path="${CHECKPOINT}" \
        adapter_checkpoint_path="${DIAGONAL_ADAPTER_CHECKPOINT}" \
        output_json="${DIAGONAL_JSON}" \
        data.source_data_path="${DATA_PATH}" \
        data.appearance_shift=medium \
        data.batch_size="${BATCH_SIZE}" \
        data.num_workers="${NUM_WORKERS}" \
        probe_train_batches="${PROBE_TRAIN_BATCHES}" \
        probe_val_batches="${PROBE_VAL_BATCHES}" \
        probe_steps="${PROBE_STEPS}" \
        rollout_batches="${ROLLOUT_BATCHES}"

echo ""
echo "[5/5] Summarizing comparison"
python scripts/summarize_medium_shift_eval.py \
    --source-json "${SOURCE_JSON}" \
    --medium-json "${MEDIUM_JSON}" \
    --delta-json "${DELTA_JSON}" \
    --diagonal-json "${DIAGONAL_JSON}" \
    --output-json "${SUMMARY_JSON}" \
    --output-csv "${SUMMARY_CSV}"

echo ""
echo "Wrote:"
echo "  ${SOURCE_JSON}"
echo "  ${MEDIUM_JSON}"
echo "  ${DELTA_JSON}"
echo "  ${DIAGONAL_JSON}"
echo "  ${SUMMARY_JSON}"
echo "  ${SUMMARY_CSV}"
