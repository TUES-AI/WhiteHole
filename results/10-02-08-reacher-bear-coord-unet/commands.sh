#!/usr/bin/env bash
set -euo pipefail
cd /proj/project
export MUJOCO_GL=egl PYTHONPATH=.
PY=/opt/venv/bin/python
DATA=/proj/data/reacher-conv/reacher_random_states.h5
VIDEO=/proj/data/reacher-conv/bear_raw_24fps.mp4
OUT=/proj/data/reacher-conv/experiment

for seed in 123 124; do
  name=coord_unet_pixel10
  [[ $seed == 124 ]] && name=${name}_seed124
  $PY scripts/train_reacher_conv_dynamics_adapter.py \
    --dataset-name "$DATA" --background-video "$VIDEO" \
    --target-variant dcs_bear_dynamic --seed "$seed" \
    --train-samples 256 --val-samples 64 --epochs 64 --batch-size 32 \
    --architecture coord_unet --base-channels 16 \
    --episode-min 0 --episode-max 127 --lr 1e-4 \
    --source-identity-weight 1 --source-pixel-weight 1 \
    --target-source-latent-weight 1 --target-source-pixel-weight 10 \
    --output-dir "$OUT/$name"
done

EXCLUDED=(138 139 140 144 151 153 175 179 183 184 185 192 193 195 197 \
  210 211 217 219 222 225 226 228 233 235 237 246 252)
EVAL=(--dataset-name "$DATA" --background-video "$VIDEO" \
  --variants source dcs_bear_dynamic --num-eval 100 --seed 4242 \
  --episode-min 128 --episode-max 255 --exclude-episodes "${EXCLUDED[@]}")
FINAL="$OUT/final_holdout"
$PY scripts/eval_reacher_shifts.py "${EVAL[@]}" \
  --output-json "$FINAL/eval_unadapted.json"
$PY scripts/eval_reacher_shifts.py "${EVAL[@]}" \
  --adapter-checkpoint /proj/data/reacher-conv/stn_only.ckpt \
  --output-json "$FINAL/eval_stn_only.json"
$PY scripts/eval_reacher_shifts.py "${EVAL[@]}" \
  --adapter-checkpoint "$OUT/coord_unet_pixel10/adapter_latest.ckpt" \
  --output-json "$FINAL/eval_coord_unet_pixel10.json"
$PY scripts/eval_reacher_shifts.py "${EVAL[@]}" \
  --adapter-checkpoint "$OUT/coord_unet_pixel10_seed124/adapter_latest.ckpt" \
  --output-json "$FINAL/eval_coord_unet_pixel10_seed124.json"
