# Reacher dynamic-camera hard benchmark

The `dynamic_camera` variant changes only rendering. It uses a deterministic
seed-derived phase and a 24-step camera orbit with varying elevation, radius,
field of view, and look-at point. Evaluation goal images are synchronized to
the live camera pose.

## Matched 30-start result

| Condition | Success |
|---|---:|
| Unadapted LeWM | 3/30 (10.0%) |
| Fixed-camera STN zero-shot | 4/30 (13.3%) |
| Dynamic-trained STN-only | 4/30 (13.3%) |

The dynamic-trained STN used 256 training transitions, 64 validation
transitions, and 512 optimizer updates. Its held-out dynamics MSE changed from
`0.467631` to `0.288318`. All evaluator protocol fields and sampled starts are
identical across the three JSON files.

## Visibility diagnostic

- Frames: 50
- Orange arm pixels: minimum 135, median 304.5, maximum 576
- Minimum nonblack image fraction: 0.8379
- Arm frames touching a five-pixel image border: 0

The contact sheet is `reacher_shift_contact_sheet.png`. Generated model
checkpoints remain under the ignored local experiment directory.

## Camera-calibrated projective oracle

The evaluator also supports two camera-calibrated rectification conditions.
`dynamic_camera_homography` maps the Reacher workspace plane at `z=0.015`
from the instantaneous dynamic camera into the source view using the known
MuJoCo extrinsics and field of view. `dynamic_camera_oracle` applies the same
homography and fills pixels outside the dynamic view with a source workspace
background rendered with all movable geoms hidden. Neither condition updates
LeWM or renders the current arm or target from the source camera.

| Evaluation on matched starts | 30 cases | 100 cases |
|---|---:|---:|
| Source LeWM | 22/30 (73.3%) | 85/100 (85%) |
| Raw dynamic camera | 3/30 (10.0%) | 12/100 (12%) |
| Calibrated homography | 22/30 (73.3%) | **64/100 (64%)** |
| Homography + static completion | 20/30 (66.7%) | **64/100 (64%)** |

On 100 paired starts, homography improves over raw dynamic camera by 52
percentage points (95% paired bootstrap CI +41 to +63). It gains on 55 raw
failures and loses 3 raw successes (exact McNemar `p=2.26e-13`). Homography
remains 21 points below source (CI -33 to -9; `p=0.00246`). Static completion
and pure homography differ on 16 starts, split 8/8 (`p=1.0`), so completion
does not improve control despite reducing the six-phase paired pixel MSE from
`983.2` to `870.2`.

The result identifies projective camera canonicalization as the right first
mechanism and shows that marginal pixel similarity is not the remaining
bottleneck. The next learned experiment should predict the eight-degree-of-
freedom homography from dynamic observations, first with simulator pose labels
as a supervised upper bound and then from short temporal windows with frozen
LeWM dynamics and inverse-dynamics supervision.

Artifacts:

- `oracle_homography_eval_30.json`
- `unadapted_eval_100.json`
- `homography_eval_100.json`
- `oracle_completion_eval_100.json`
- `oracle_rectification_contact_sheet.png`
- `oracle_rectification_diagnostics.json`
