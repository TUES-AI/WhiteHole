#!/usr/bin/env bash
set -euo pipefail
cd /proj/project
export MUJOCO_GL=egl PYTHONPATH=.
PY=/opt/venv/bin/python
DATA=/proj/data/reacher-dcs-bear/reacher_random_states.h5
VIDEO=/proj/data/reacher-dcs-bear/bear_raw_24fps.mp4
OUT=/proj/data/reacher-dcs-bear/experiment-v2

$PY scripts/verify_reacher_dcs_background.py \
  --background-video "$VIDEO" --output-dir /proj/data/reacher-dcs-bear/verification-v2

TRAIN=(--dataset-name "$DATA" --background-video "$VIDEO" \
  --target-variant dcs_bear_dynamic --seed 123 --train-samples 256 \
  --val-samples 64 --epochs 64 --batch-size 32 \
  --episode-min 0 --episode-max 127)
$PY scripts/train_reacher_movie_adapter.py "${TRAIN[@]}" --output-dir "$OUT/full"
$PY scripts/train_reacher_movie_adapter.py "${TRAIN[@]}" --output-dir "$OUT/stn_only" \
  --freeze-encoder --freeze-projector
$PY scripts/train_reacher_movie_adapter.py "${TRAIN[@]}" --output-dir "$OUT/stn_encoder" \
  --freeze-projector

EVAL=(--dataset-name "$DATA" --background-video "$VIDEO" \
  --variants source dcs_bear_dynamic --num-eval 100 --seed 42 \
  --episode-min 128 --episode-max 255)
$PY scripts/eval_reacher_shifts.py "${EVAL[@]}" \
  --output-json "$OUT/eval_unadapted.json"
for name in full stn_only stn_encoder; do
  $PY scripts/eval_reacher_shifts.py "${EVAL[@]}" \
    --adapter-checkpoint "$OUT/$name/movie_latest.ckpt" \
    --output-json "$OUT/eval_$name.json"
done
