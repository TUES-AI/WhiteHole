#!/bin/bash
# Submit a sequential sweep for delta-vector adapters with source-frame anchors.
#
# Run from /valhalla/projects/bg-eng-01/WhiteHole:
#   bash slurm/whitehole/06_submit_delta_pair_anchor_sweep.sh
#
# Override weights with:
#   PAIR_ALIGNMENT_WEIGHTS="0.01 0.03 0.1 0.3" bash ...

set -euo pipefail

CONFIG="${CONFIG:-configs/adaptation/two_rooms_medium_delta_proposal.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/pldm/two_rooms_jepa_baseline_len17_3m/epoch=10_sample_step=2072576.ckpt}"
DATA_PATH="${DATA_PATH:-outputs/data/two_rooms_len17_3m.npz}"
APPEARANCE_SHIFT="${APPEARANCE_SHIFT:-medium}"
PAIR_ALIGNMENT_WEIGHTS="${PAIR_ALIGNMENT_WEIGHTS:-0.01 0.03 0.1 0.3}"
EPOCHS="${EPOCHS:-3}"
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
IDENTITY_PRIOR_WEIGHT="${IDENTITY_PRIOR_WEIGHT:-0.0001}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-slurm/whitehole/03_train_appearance_adapter.sh}"
EVAL_SCRIPT="${EVAL_SCRIPT:-slurm/whitehole/04_eval_appearance_adapter.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/adaptation}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-outputs/eval}"
CHAIN="${CHAIN:-1}"
TRAIN_TIME="${TRAIN_TIME:-00:20:00}"
EVAL_TIME="${EVAL_TIME:-00:10:00}"

previous_job=""

for weight in ${PAIR_ALIGNMENT_WEIGHTS}; do
    tag="${weight//./d}"
    tag="${tag//-/m}"
    run_name="two_rooms_medium_delta_pairw_${tag}_3ep"
    output_dir="${OUTPUT_ROOT}/${run_name}"
    output_json="${EVAL_OUTPUT_ROOT}/${run_name}_adapter_eval.json"

    dependency_args=()
    if [ "${CHAIN}" = "1" ] || [ "${CHAIN}" = "true" ]; then
        if [ -n "${previous_job}" ]; then
            dependency_args=(--dependency="afterok:${previous_job}")
        fi
    fi

    train_id=$(
        CONFIG="${CONFIG}" \
        CHECKPOINT="${CHECKPOINT}" \
        DATA_PATH="${DATA_PATH}" \
        OUTPUT_DIR="${output_dir}" \
        APPEARANCE_SHIFT="${APPEARANCE_SHIFT}" \
        EPOCHS="${EPOCHS}" \
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
        IDENTITY_PRIOR_WEIGHT="${IDENTITY_PRIOR_WEIGHT}" \
        PAIR_ALIGNMENT_WEIGHT="${weight}" \
        AUTO_EVAL=0 \
        sbatch --parsable --time="${TRAIN_TIME}" "${dependency_args[@]}" "${TRAIN_SCRIPT}"
    )

    eval_id=$(
        CONFIG="${CONFIG}" \
        CHECKPOINT="${CHECKPOINT}" \
        ADAPTER_CHECKPOINT="${output_dir}/adapter_latest.ckpt" \
        DATA_PATH="${DATA_PATH}" \
        APPEARANCE_SHIFT="${APPEARANCE_SHIFT}" \
        OUTPUT_JSON="${output_json}" \
        BATCH_SIZE="${BATCH_SIZE}" \
        NUM_WORKERS="${NUM_WORKERS}" \
        sbatch --parsable --time="${EVAL_TIME}" --dependency="afterok:${train_id}" "${EVAL_SCRIPT}"
    )

    printf "pair_alignment_weight=%s train_job=%s eval_job=%s output_dir=%s eval_json=%s\n" \
        "${weight}" "${train_id}" "${eval_id}" "${output_dir}" "${output_json}"

    previous_job="${eval_id}"
done
