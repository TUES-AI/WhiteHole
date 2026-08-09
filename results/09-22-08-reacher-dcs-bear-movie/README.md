# Reacher DAVIS bear background: MoVie-style scope ablation

## Result

A dynamic DAVIS `bear` video reduced unadapted closed-loop success from 82% on source to 18%. None of the three adapted models improved bear control, although every adapter lowered held-out one-step dynamics MSE. All three also catastrophically damaged source control.

| Condition | Trainable params | Bear val dynamics MSE | Source success | Bear dynamic success |
|---|---:|---:|---:|---:|
| Unadapted LeWM | 0 | 0.2731 | **82/100** | **18/100** |
| STNs + encoder + projector | 6,377,516 | **0.1169** | **13/100** | 9/100 |
| STNs only | 83,372 | 0.1975 | 6/100 | 16/100 |
| STNs + encoder, projector frozen | 5,584,748 | 0.1382 | **13/100** | 8/100 |

Against unadapted bear control, paired differences were -9 points for the full update (95% bootstrap CI -18 to 0; exact McNemar p=0.093), -2 for STN-only (-11 to +7; p=0.832), and -10 for STNs+encoder (-19 to -1; p=0.052). The result supports insufficiency for this tested shift and budget, not a universal impossibility claim.

## Protocol

- DAVIS 2017 `bear_raw_24fps.mp4`, 82 frames at 854x480 and 24 FPS.
- SHA-256: `46f05e52f884a5180cd44ee0448e461fc744f2665e0ae5215e2d52ba1fac28bd`.
- DCS-style MuJoCo sky-texture replacement, transparent Reacher floor, deterministic bidirectional playback.
- 256 training and 64 validation transitions from episodes 0-127, seed 123, 512 optimizer updates, batch size 32.
- One-step frozen-dynamics MSE only; dynamics, action encoder, and control core frozen.
- 100 matched evaluation starts from disjoint episodes 128-255, seed 42, 25-step goals, 50-action budget, CEM horizon 5, 300 samples, 30 iterations, top-k 30.
- One adaptation seed and offline fixed-buffer adaptation; dynamic background only.

The original `dmc/reacher_random` dataset was not publicly accessible, so all rows use the same newly collected compact dataset: 256 random-policy episodes of 200 steps, collection seed 3072, SHA-256 `5d405963441f161fbede306996dda85bc5f69a1f69c1cca5be762ab32f4ed5e2`. This makes the rows internally matched but not an exact rerun of the collaborator's hard-camera starts.

## Verification and artifacts

`verification/pixel_only_check.png` shows identical Reacher states over source and consecutive bear frames. `verification.json` confirms unchanged qpos/qvel, exact repeated-frame determinism after renderer warm-up, and temporal background change.

- `summary.json`: consolidated metrics, paired tests, hashes, and limitations.
- `full/`, `stn_only/`, `stn_encoder/`: training reports and logs.
- `eval_*.json`: per-case matched control outcomes.
- `commands.sh`: exact experiment commands.
- `environment.txt`: software and GPU metadata.

Checkpoints are excluded from Git and preserved at `/Volumes/SSD/whitehole/artifacts/09-22-08-reacher-dcs-bear-movie/experiment-v2/`.
