07-20-08 | Push-T official DINO-WM smoke integration
Experiment description: Ran the official external DINO-WM Push-T stack with the official checkpoint and dataset. Code commit: fdccccd. Result folder: results/07-20-08-pusht-official-smoke/.
Experiment results: The pipeline completed successfully (job 186307, runtime 00:00:59), but the intentionally tiny planner run achieved success_rate 0.0. This is an execution smoke test, not a benchmark or adaptation result.

---

09-19-08 | Reacher medium full-encoder distribution alignment
Experiment description: Fine-tuned the full 5,501,376-parameter LeWM Reacher encoder on paired source/medium observations while keeping the predictor and control core frozen. The objective combined paired latent alignment (1.0), source dynamics alignment (0.25), latent SWD (0.05), transition SWD (0.1), joint state-action-transition SWD (0.1), and source identity (0.05). Code: `scripts/train_reacher_medium_encoder_adapter.py` and `scripts/eval_reacher_shifts.py`; this is a preserved local experiment with no dedicated commit. Result folder: `tmp_reacher_visualization/medium_encoder_adapter/full_joint_swd_continuation_lr1e5/`.
Experiment results: medium visual-shift control success was **15/30 (50%)** on the original evaluation and **41/100 (41%)** on the larger follow-up. This is a single training seed and a paired-supervision exploratory result, not a MoVie reproduction or final benchmark.

---

09-20-08 | Reacher hard-camera offline MoVie-style baseline
Experiment description: Adapted identity-initialized affine STNs at RGB input and the ViT patch-projection grid, plus the LeWM visual encoder/projector at learning rate `1e-7`, using only target-view one-step frozen-dynamics MSE. The STNs used learning rate `1e-5`; dynamics, action encoder, and control remained frozen. Training used 256 target transitions, batch size 32, and 512 optimizer updates. Code: `scripts/reacher_movie_adapter.py`, `scripts/train_reacher_movie_adapter.py`, and `scripts/eval_reacher_shifts.py`; no dedicated commit yet. Result folder: `tmp_reacher_visualization/movie_adapter/hard_camera/`.

| Evaluation on 30 matched starts | Unadapted | MoVie-style | Paired exact test |
|---|---:|---:|---:|
| Hard camera | 6/30 (20.0%) | **21/30 (70.0%)** | `p=0.00073` |
| Source retention | **22/30 (73.3%)** | 21/30 (70.0%) | `p=1.0` |

Held-out target dynamics MSE decreased from `0.2664` to `0.0741`; an unused paired diagnostic decreased from `1.1275` to `0.1361`. This is one training seed and offline pre-adaptation from a fixed target buffer, not the paper's online interaction-coupled protocol.

---

09-21-08 | Reacher hard-camera MoVie 100-case evaluation
Experiment description: Re-evaluated the fixed MoVie-style checkpoint and unadapted LeWM on 100 matched dataset-derived starts under both source and hard-camera views. Planner and environment protocol were unchanged: 25-step goals, 50-action budget, CEM horizon 5, 300 samples, 30 iterations, and top-k 30. Code: `scripts/eval_reacher_shifts.py`; no dedicated commit yet. Result files: `tmp_reacher_visualization/movie_adapter/hard_camera/unadapted_source_hard_success_rates_100.json` and `tmp_reacher_visualization/movie_adapter/hard_camera/movie_source_hard_success_rates_100.json`.

| Evaluation on 100 matched starts | Unadapted | MoVie-style | Paired difference (95% bootstrap CI) |
|---|---:|---:|---:|
| Hard camera | 35/100 (35%) | **75/100 (75%)** | **+40 pp** (`+28` to `+52`), exact `p=2.3e-8` |
| Source retention | **85/100 (85%)** | 78/100 (78%) | `-7 pp` (`-17` to `+3`), exact `p=0.248` |

MoVie fixed 47 hard-camera failures and regressed on 7 prior hard-camera successes. It recovered 80% of the 50-point source-to-hard gap. Adapted hard versus original source retained a 10-point aggregate gap (`p=0.099`), while adapted hard versus adapted source differed by only 3 points (`p=0.728`). The larger evaluation supersedes the 30-case pilot as the primary point estimate, but it still uses one trained checkpoint and does not measure training-seed uncertainty.

---

09-21-08 | Reacher hard-camera MoVie scope and preservation ablation
Experiment description: Compared five MoVie-style visual-adaptation conditions using the same hard-camera target buffer, 256 training transitions, 64 validation transitions, batch size 32, 512 optimizer updates, seed 123, and frozen LeWM dynamics/control core. Conditions were: STNs only; STNs plus ViT encoder with projector frozen; the existing STNs plus encoder and projector baseline; the full update plus source-identity MSE at weight `0.05`; and the full update plus source-relative latent, predicted-transition, and joint sliced-Wasserstein preservation at weights `0.05/0.1/0.1`. No target/source pairwise MSE was used. Every checkpoint was evaluated on the same 100 source and hard-camera starts with the unchanged full CEM protocol. Code: `scripts/train_reacher_movie_adapter.py`, `scripts/reacher_movie_adapter.py`, and `scripts/eval_reacher_shifts.py`. Results: `tmp_reacher_visualization/movie_adapter/ablation/`.

| Condition | Trainable parameters | Val dynamics MSE | Source | Hard camera |
|---|---:|---:|---:|---:|
| Unadapted LeWM | 0 | `0.2664` | **85/100** | 35/100 |
| STNs + encoder + projector | 6,377,516 | `0.0741` | 78/100 | 75/100 |
| **STNs only** | **83,372** | `0.0876` | 77/100 | **82/100** |
| STNs + encoder, projector frozen | 5,584,748 | `0.0755` | 80/100 | 73/100 |
| Full update + source identity | 6,377,516 | `0.0706` | 83/100 | 70/100 |
| Full update + distribution SWD | 6,377,516 | **`0.0701`** | 80/100 | 78/100 |

All new adaptations significantly improved hard-camera success over unadapted LeWM. STN-only gained 47 points (paired 95% bootstrap CI +36 to +58; exact McNemar `p=1.18e-12`). Updating the encoder with the projector frozen scored 9 points below STN-only on hard camera (CI -17 to -1; `p=0.049`), even though its validation dynamics MSE was lower. Source identity retained source performance within 2 points of unadapted LeWM (`p=0.839`) but scored 12 points below STN-only on hard camera (CI -22 to -2; `p=0.029`). Distribution SWD scored 4 points below STN-only on hard camera (CI -13 to +5; `p=0.503`) and 3 points above the unregularized full update (CI -7 to +14; `p=0.711`), so those differences are unresolved at 100 starts.

The distribution objective reduced held-out latent SWD from `0.5415` to `0.0477`, predicted-transition SWD from `0.3786` to `0.0572`, and joint state/action/transition SWD from `0.3298` to `0.0349`. Source-retention latent MSE was `0.1559` for STN-only, `0.1283` for the unregularized full update, `0.0659` with source identity, and `0.1136` with SWD. Frozen-core gradient tensors were zero in every condition. The downstream ranking does not follow validation dynamics MSE, reinforcing that one-step fit and distribution alignment are diagnostics rather than substitutes for control evaluation. Limitations: one training seed, one identity/SWD weight setting, offline pre-adaptation, and one hard-camera shift.

---

09-22-08 | Reacher dynamic DAVIS bear MoVie-style scope ablation
Experiment description: Replaced Reacher's MuJoCo sky texture with deterministic dynamic DAVIS 2017 `bear` frames and made its floor transparent, following the Distracting Control Suite background mechanism without its TensorFlow reader. Compared unadapted LeWM, STNs only, STNs plus encoder with projector frozen, and STNs plus encoder and projector. Adaptation used 256 transitions and 64 validation transitions from episodes 0-127, seed 123, batch size 32, 512 updates, and one-step frozen-dynamics MSE; dynamics, action encoder, and control remained frozen. All rows used 100 matched evaluation starts from disjoint episodes 128-255 and the full CEM protocol. The unavailable private `dmc/reacher_random` cache was replaced by one shared 256-episode random-policy state/action dataset. Code was executed from uncommitted changes layered on `3279d92`; result folder: `results/09-22-08-reacher-dcs-bear-movie/`.

| Condition | Trainable parameters | Bear val dynamics MSE | Source | Dynamic bear |
|---|---:|---:|---:|---:|
| Unadapted LeWM | 0 | `0.2731` | **82/100** | **18/100** |
| STNs + encoder + projector | 6,377,516 | **`0.1169`** | **13/100** | 9/100 |
| **STNs only** | **83,372** | `0.1975` | 6/100 | 16/100 |
| STNs + encoder, projector frozen | 5,584,748 | `0.1382` | **13/100** | 8/100 |

The dynamic bear background created a 64-point unadapted source-to-target drop. Relative to unadapted bear control, the full, STN-only, and STN+encoder rows changed success by -9 points (95% paired bootstrap CI -18 to 0; exact `p=0.093`), -2 (-11 to +7; `p=0.832`), and -10 (-19 to -1; `p=0.052`). Every adaptation lowered bear one-step MSE but failed to recover control and reduced source success by 69-76 points. Frozen-core gradient tensors were zero. This is a difficult negative result for the tested offline MoVie-style adapters and budget, not proof that all MoVie-derived methods must fail. Limitations: one training seed, one dynamic video, no static-background row, and a new internally matched dataset rather than the collaborator's private cache.

---

10-02-08 | Reacher dynamic bear coordinate-U-Net upper bound
Experiment description: Trained a 97,731-parameter identity-initialized coordinate-aware residual U-Net before fully frozen LeWM perception/dynamics/control. Training used episodes 0-127, 256 transitions, 64 validation transitions, 512 updates, and seeds 123/124. Screening showed that target-only dynamics consistency and source preservation did not recover bear control; the successful objective adds exact paired source-latent alignment and source-pixel reconstruction at weights 1 and 10, plus source latent/pixel identity at weights 1/1. Final evaluation used 100 matched starts after excluding all 28 episode IDs encountered during hyperparameter screening. Code was executed from uncommitted changes layered on `2be2600`; result folder: `results/10-02-08-reacher-bear-coord-unet/`.

| Condition | Parameters | Bear val dynamics MSE | Source | Dynamic bear |
|---|---:|---:|---:|---:|
| Unadapted LeWM | 0 | — | 73/100 | 19/100 |
| STN-only | 83,372 | — | 16/100 | 10/100 |
| Coordinate U-Net, seed 123 | 97,731 | `0.1544` | 80/100 | 29/100 |
| Coordinate U-Net, seed 124 | 97,731 | `0.1762` | **86/100** | **35/100** |

Seed 123 improved bear success over STN-only by 19 points (95% paired bootstrap CI +8 to +30; exact `p=0.0013`). Seed 124 improved it by 25 points (+14 to +36; `p=7.0e-5`) and beat unadapted bear success by 16 points (+4 to +28; `p=0.0139`). Two-seed means were 83% source and 32% bear. Frozen-core gradient tensors were zero. This establishes a small non-affine adapter capacity upper bound, not a target-only adaptation solution: the successful loss requires exact source renders and strong pixel supervision unavailable in ordinary deployment. The target-only 83,052-parameter residual CNN screened at 3/30 bear success despite very low dynamics MSE.
---

09-22-08 | Reacher dynamic-camera hard benchmark
Experiment description: Added a `dynamic_camera` observation shift that completes a camera orbit every 24 environment steps while smoothly varying radius (`0.20`-`0.28`), height (`0.64`-`0.84`), field of view (`40`-`52` degrees), and look-at offset. Each episode receives a deterministic seed-derived phase. Physics and target state are unchanged, and the evaluator rerenders the goal at the same instantaneous camera as the live observation. Trained an STN-only MoVie adapter with the fixed-camera protocol: 83,372 trainable parameters, 256 target transitions, 64 validation transitions, batch size 32, 512 updates, and frozen encoder/projector/dynamics/control. Code: `scripts/visualize_reacher_shifts.py`, `scripts/eval_reacher_shifts.py`, and `scripts/train_reacher_movie_adapter.py`. Results: `results/09-22-08-reacher-dynamic-camera/`.

| Dynamic-camera evaluation on 30 matched starts | Success |
|---|---:|
| Unadapted LeWM | 3/30 (10.0%) |
| Fixed-hard-camera STN, zero-shot | 4/30 (13.3%) |
| Dynamic-camera-trained STN | 4/30 (13.3%) |

The fixed-camera and dynamic-trained STNs each differ from unadapted LeWM by only one success (paired exact `p=1.0`), and they differ from each other on two discordant starts with equal aggregate success (`p=1.0`). Dynamic-camera STN training reduced held-out dynamics MSE from `0.4676` to `0.2883`, far less than the same STN budget's `0.2664` to `0.0876` reduction on the fixed hard camera. In the 50-frame visual diagnostic, orange arm pixels were detected in every frame (`135` minimum), at least 83.8% of pixels were nonblack, and the arm never touched a five-pixel image border. This is therefore a difficult visible observation-dynamics shift rather than an occlusion failure. Limitations: 30 control starts, one training seed, and one dynamic-camera schedule; this establishes a stress-test candidate, not a final benchmark estimate.

---

09-23-08 | Reacher calibrated projective canonicalization oracle
Experiment description: Used known MuJoCo camera extrinsics and vertical field of view to derive an eight-degree-of-freedom homography from the instantaneous dynamic-camera workspace plane (`z=0.015`) into LeWM's source camera. LeWM and its planner remained frozen. A second condition filled pixels outside the dynamic view with a source workspace background rendered after hiding every movable geom, so it contained no current arm or target state. Live and goal frames used the same synchronized camera transform. Code: `scripts/visualize_reacher_shifts.py`, `scripts/eval_reacher_shifts.py`, and `tests/test_reacher_visual_shifts.py`. Results: `results/09-22-08-reacher-dynamic-camera/`.

| Evaluation on 100 matched starts | Success |
|---|---:|
| Source LeWM | 85/100 (85%) |
| Raw dynamic camera | 12/100 (12%) |
| Calibrated homography | **64/100 (64%)** |
| Homography + static completion | **64/100 (64%)** |

The homography improves over raw dynamic camera by 52 percentage points (paired 95% bootstrap CI +41 to +63), gaining on 55 raw failures and losing 3 raw successes (exact McNemar `p=2.26e-13`). It remains 21 points below source (CI -33 to -9; `p=0.00246`). Static completion and pure homography swap 8 successes in each direction (`p=1.0`), despite completion reducing the six-phase paired pixel MSE from `983.2` to `870.2`. This supports projective canonicalization as the primary mechanism, rejects static background completion as the explanation for the remaining gap, and again shows that pixel-level fit need not predict control. The immediate learned experiment is an image-conditioned projective transformer supervised by simulator camera pose, followed by a short-window temporal version without pose labels. Limitations: the oracle knows exact camera calibration, assumes an approximately planar workspace, and still loses information when source-view pixels are outside the dynamic camera.
---

11-17-08 | Frozen I-JEPA and V-JEPA 2.1 structured visual-adapter screen
Experiment description: Screened a 97,731-parameter coordinate-aware residual U-Net and a 67,009-parameter grid/color rectifier before frozen JEPA backbones. I-JEPA used seed `123`, the official ImageNet-1K ViT-H/14 checkpoint (`0382013...122`), and 20 genuine ImageNet classes from Imagenette + Imagewoof, with disjoint probe/adaptation/source-prior/capacity/final subsets of 2,000/2,000/2,000/400/1,000 images. The fixed domains were RGB to RBG, +30 degree rotation with +18/+8 degree x/y shear and reflection padding, and their composition. Paired capacity used 512 updates, batch 64, learning rate `3e-4`, and equal clean-source identity weight; masked-I-JEPA rows used 256 updates, batch 8, and `1e-4`. A condition passed only after recovering at least half the pixel gap on the separate capacity split. Code: `scripts/jepa_visual_adapters.py`, `scripts/train_ijepa_visual_adapters.py`, and `scripts/train_vjepa_visual_adapters.py`. Results: `results/11-17-08-jepa-visual-adapters/`; full checkpoints: `/Volumes/SSD/whitehole/artifacts/11-jepa-visual-adapters/`.

| I-JEPA condition | Objective | Source probe | Target probe | JEPA loss | Pixel L1 | Capacity recovery |
|---|---|---:|---:|---:|---:|---:|
| Unadapted source | — | 95.0% | 95.0% | — | — | — |
| Unadapted RBG | — | — | 92.7% | `0.19170` | `0.05514` | — |
| Unadapted fixed affine | — | — | 90.4% | `0.20210` | `0.18238` | — |
| Unadapted fixed composed | — | — | 78.3% | `0.21670` | `0.20059` | — |
| U-Net RBG | Paired | 94.7% | **93.4%** | `0.18940` | `0.02113` | **61.7%, pass** |
| U-Net RBG | Target-only | 95.1% | 92.7% | `0.19111` | `0.05893` | — |
| U-Net RBG | Target + source identity | 95.0% | 92.7% | `0.19165` | `0.05528` | — |
| U-Net affine | Paired | 95.0% | 90.4% | `0.20205` | `0.18237` | **0.004%, fail** |
| U-Net composed | Paired | 94.9% | 82.2% | `0.21558` | `0.18687` | **6.9%, fail** |
| Grid/color RBG | Paired | 93.8% | **93.6%** | `0.19028` | `0.01944` | **64.3%, pass** |
| Grid/color RBG | Target-only | 94.8% | 92.7% | `0.19268` | `0.10289` | — |
| Grid/color RBG | Target + source identity | 94.9% | 92.7% | `0.19150` | `0.05507` | — |
| Grid/color affine | Paired | 95.0% | 90.5% | `0.20195` | `0.18235` | **0.017%, fail** |
| Grid/color composed | Paired | 95.0% | 83.7% | `0.21460` | `0.19024` | **5.2%, fail** |

Target-only and target-plus-source-identity geometry/composed rows were skipped after their paired gates failed. A prior developmental run varied affine sign and magnitude by hidden image index and did not define one coherent target convention. Its exact pre-fix code snapshot and command were not preserved, so the archived report is a non-reproducible protocol lesson and none of its metrics support the primary result. The fixed-domain adapter optimization loops totaled 987 seconds; model loading, probing, and evaluation were not included. A supplementary real-checkpoint MPS update and a logged A6000 CUDA update both propagated finite nonzero gradients only to the adapter; the MPS command was not preserved and is not primary evidence. The CUDA update used 2,845,008,384 peak bytes, loss `0.13184`, gradient norm `0.14204`, and maximum update `1.0e-4`. Final frozen-core gradient tensors and sentinel drift were zero. Exact primary invocations are preserved in `commands.sh` because the training logs do not echo their shell commands.

V-JEPA used seed `20260811`, the official V-JEPA 2.1 ViT-B student/predictor (`848a77...f4d`), and ViT-G target encoder (`7aae1a...58a`) on five balanced `nateraw/kinetics-mini` classes. Disjoint probe/adaptation/source-prior/capacity/final subsets contained 25/15/10/25/25 videos; every clip used 16 centered frames at 4 FPS and 384-pixel crops. The framewise RBG shift produced no observed final-probe drop: source and RBG both scored 22/25 (88%). After 512 paired updates at batch one, U-Net scored 22/25 source and 21/25 target with 29.1% capacity pixel-gap recovery; grid/color scored 22/25 source and target with 38.1% recovery. Both failed the 50% gate, so target-only and target-plus-source-identity rows were skipped. Their class-balanced five-clip V-JEPA losses (`0.58186` and `0.58175`) did not improve on unadapted (`0.58113`) and did not imply downstream recovery. A separate faithful ViT-B/predictor/ViT-G target-loss update succeeded in 3.29 seconds with loss `0.58523`, adapter gradient norm `0.21627`, maximum update `1.0e-4`, zero frozen gradients, and 12,875,506,176 peak CUDA bytes; its exact command and stdout are preserved. Within each class, lexicographically sorted filenames were partitioned without random shuffling, so this tiny split is ordering-dependent. The video result is therefore inconclusive downstream and negative only for paired capacity under this adapter/budget. Full-run sentinel drift was zero.

The decisive result is negative: target-only masked-I-JEPA did not improve RBG probe accuracy, neither small adapter could repair fixed I-JEPA geometry under paired supervision, and the tiny V-JEPA probe exposed no RBG accuracy gap while both paired adapters failed capacity. Lower JEPA loss did not rank downstream recovery. This does not establish that JEPA adaptation is impossible; the next image adapter needs stronger global spatial capability, and the next video screen needs a larger probe-valid subset before target-only optimization is informative.

---
