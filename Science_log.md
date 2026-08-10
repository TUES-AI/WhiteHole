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
