# shellcheck shell=bash
# Chain WhiteHole epoch-checkpoint jobs across a 4h Slurm wall-time cap.
#
# Required caller variables:
#   OUTPUT_ROOT   root checkpoint directory
#   OUTPUT_DIR    run-specific checkpoint directory under OUTPUT_ROOT
#   EPOCHS        target epoch value passed to whitehole.train
#   SELF_SCRIPT   repo-relative Slurm script path to resubmit
#
# Optional:
#   AUTO_CHAIN=0   disable successor submission
#   CHAIN_MAX=20   cap chain length

: "${OUTPUT_ROOT:?_autochain_epochs.sh requires OUTPUT_ROOT}"
: "${OUTPUT_DIR:?_autochain_epochs.sh requires OUTPUT_DIR}"
: "${EPOCHS:?_autochain_epochs.sh requires EPOCHS}"
: "${SELF_SCRIPT:?_autochain_epochs.sh requires SELF_SCRIPT}"

_ac_ckpt_dir="${OUTPUT_ROOT%/}/${OUTPUT_DIR#/}"
_ac_latest_epoch=-1

if [ -d "${_ac_ckpt_dir}" ]; then
    for _ac_p in "${_ac_ckpt_dir}"/epoch=*_sample_step=*.ckpt; do
        [ -f "${_ac_p}" ] || continue
        _ac_b="$(basename "${_ac_p}")"
        _ac_e="${_ac_b#epoch=}"
        _ac_e="${_ac_e%%_sample_step=*}"
        case "${_ac_e}" in ''|*[!0-9]*) continue ;; esac
        _ac_e=$((10#${_ac_e}))
        if [ "${_ac_e}" -gt "${_ac_latest_epoch}" ]; then
            _ac_latest_epoch="${_ac_e}"
        fi
    done
fi

if [ "${_ac_latest_epoch}" -ge "${EPOCHS}" ]; then
    echo "[autochain] target reached: latest checkpoint epoch ${_ac_latest_epoch} >= EPOCHS ${EPOCHS}."
    echo "[autochain] nothing to do; exiting without queuing a successor."
    exit 0
fi

if [ "${_ac_latest_epoch}" -ge 0 ]; then
    echo "[autochain] latest checkpoint epoch ${_ac_latest_epoch}; whitehole.train will resume from ${_ac_ckpt_dir}."
else
    echo "[autochain] no checkpoint found in ${_ac_ckpt_dir}; starting fresh."
fi

CHAIN_DEPTH="${CHAIN_DEPTH:-0}"
CHAIN_MAX="${CHAIN_MAX:-20}"

if [ "${AUTO_CHAIN:-1}" != "1" ]; then
    echo "[autochain] AUTO_CHAIN disabled; running a single standalone job."
elif [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "[autochain] not under SLURM; running once without chaining."
elif [ "${CHAIN_DEPTH}" -ge "${CHAIN_MAX}" ]; then
    echo "[autochain] CHAIN_DEPTH ${CHAIN_DEPTH} hit CHAIN_MAX ${CHAIN_MAX}; not queuing a successor."
else
    _ac_next=$((CHAIN_DEPTH + 1))
    _ac_nid="$(sbatch --parsable \
        --dependency="afterany:${SLURM_JOB_ID}" \
        --kill-on-invalid-dep=yes \
        --export="ALL,CHAIN_DEPTH=${_ac_next},MASTER_PORT=" \
        "${SELF_SCRIPT}" 2>/dev/null || true)"
    if [ -n "${_ac_nid}" ]; then
        echo "[autochain] queued successor job ${_ac_nid} (link ${_ac_next}/${CHAIN_MAX}, afterany:${SLURM_JOB_ID})"
    else
        echo "[autochain] WARNING: failed to queue successor job."
    fi
fi
