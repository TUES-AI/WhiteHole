# Learning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models

**Paper:** Vlad Sobal, Wancong Zhang, Kyunghyun Cho, Randall Balestriero, Tim G. J. Rudner, and Yann LeCun, 2025.  
**Version reviewed:** [arXiv:2502.14819v4](https://arxiv.org/abs/2502.14819v4), updated 2025-10-29; [TeX source](https://arxiv.org/src/2502.14819v4).  
**Project/code:** [project page](https://latent-planning.github.io/) · [official repository](https://github.com/vladisai/PLDM) · inspected repository commit [`1bd7e56`](https://github.com/vladisai/PLDM/tree/1bd7e564ecd961205bc18b23067b19e9ca24ac90) (2025-10-29).

## Problem and core idea

The paper asks which methods make best use of **offline state-action trajectories with no rewards** when downstream tasks and, potentially, environment layouts are not known during training. It compares five model-free methods—GCBC, CRL, GCIQL, HIQL, and HILP—with **Planning with a Latent Dynamics Model (PLDM)**.

PLDM trains an action-conditioned latent predictor with a reconstruction-free JEPA objective, then solves downstream tasks by searching over action sequences with model-predictive control (MPC). For goal reaching, the planner minimizes distance between predicted latents and the encoded goal; for state avoidance, the paper reverses that cost. The central empirical case is not merely that PLDM predicts future embeddings, but that its predictions support successful closed-loop planning under scarce, short, random, or layout-diverse offline data.

This is relevant to WhiteHole as evidence for an **evaluation principle** and as a source of controlled navigation environments. PLDM should not replace WhiteHole's frozen-source-model adaptation approach: the paper trains its world model end to end, whereas WhiteHole studies whether a frozen source encoder/predictor can remain useful after a visual observation shift through a small adapter.

## Method details (short)

Given a trajectory \((s_0,a_0,s_1,\ldots,a_{T-1},s_T)\), PLDM uses an encoder \(h_\theta\) and one or more action-conditioned predictors \(f_\theta^k\):

\[
z_t=h_\theta(s_t), \qquad \hat z_0^k=z_0, \qquad
\hat z_t^k=f_\theta^k(\hat z_{t-1}^k,a_{t-1}).
\]

Training combines:

- **latent prediction:** multi-step squared error between \(\hat z_t^k\) and \(z_t\);
- **variance and covariance regularization:** VICReg-style anti-collapse terms;
- **temporal smoothness:** nearby encoded states are encouraged to remain nearby;
- **inverse dynamics (IDM):** an MLP predicts \(a_t\) from \((z_t,z_{t+1})\).

The complete objective is

\[
\mathcal L_{\mathrm{PLDM}}=\mathcal L_{\mathrm{sim}}
+\alpha\mathcal L_{\mathrm{var}}
+\beta\mathcal L_{\mathrm{cov}}
+\delta\mathcal L_{\mathrm{time\text{-}sim}}
+\omega\mathcal L_{\mathrm{IDM}}.
\]

At test time, MPPI searches for actions that minimize latent goal distance. With an ensemble, the planner also penalizes predictor disagreement as an estimate of out-of-distribution uncertainty. MPC normally replans after every environment interaction. The paper's default prediction horizon is 16; its appendix reports five predictors for Two-Rooms and Ant U-Maze, one for Diverse PointMaze, and 500 MPPI samples.

Architectures vary by environment rather than constituting one universal model:

- **Two-Rooms:** Impala-small image encoder and a two-layer, 512-hidden-unit GRU predictor (about 2.22M parameters).
- **Diverse PointMaze:** a spatial convolutional representation of the RGB maze concatenated with expanded velocity planes, followed by a convolutional predictor (about 53.7K parameters).
- **Ant U-Maze:** a learned 256-dimensional embedding of global \((x,y)\), raw remaining proprioception, and a three-layer MLP predictor (about 1.08M parameters).

The official implementation exposes the basic [`JEPA`](https://github.com/vladisai/PLDM/blob/1bd7e564ecd961205bc18b23067b19e9ca24ac90/pldm/models/jepa.py), [VICReg objective](https://github.com/vladisai/PLDM/blob/1bd7e564ecd961205bc18b23067b19e9ca24ac90/pldm/objectives/vicreg.py), and [MPPI planner](https://github.com/vladisai/PLDM/blob/1bd7e564ecd961205bc18b23067b19e9ca24ac90/pldm/planning/planners/mppi_planner.py).

## Key results

### Controlled Two-Rooms experiments

Two-Rooms observations are two-channel \(64\times64\) top-down images: one channel for the point agent and one for walls. Actions are two-dimensional displacements with norm at most 2.45, and goal-reaching episodes last at most 200 steps. The default offline data has 3M transitions in length-91 trajectories whose action directions follow a concentrated von Mises process; the paper separately varies trajectory length, dataset size, random-action fraction, and whether trajectories cross the doorway.

- With abundant high-quality data, all methods work: PLDM reaches **97.8 ± 0.7%** success, versus HILP 100%, GCIQL 98.0%, HIQL 96.4%, CRL 89.3%, and GCBC 86.0% (mean ± standard error over three seeds).
- PLDM and GCIQL are the most sample-efficient in the dataset-size sweep, reaching roughly 80% success with only a few thousand transitions. HILP needs more data but reaches perfect performance at scale.
- With short training trajectories, PLDM and HILP can compose longer test-time behavior; GCIQL is also strong in the revised v4 results. This supports planning-based stitching when local transitions cover the needed dynamics.
- The more severe **missing-connectivity** test is a counterexample to broad stitching claims. When no offline trajectory passes through the doorway, PLDM falls to **34.4 ± 2.7%**, while HILP reaches 100% and GCIQL 99.6%. PLDM exceeds most other goal-conditioned baselines but does not infer an unobserved transition reliably.
- On uniformly random trajectories, PLDM, HILP, and GCIQL outperform the other model-free methods. The figure supports the ordering, but does not provide a table of exact values.
- Without retraining, reversing the latent cost lets PLDM avoid a pursuing agent more successfully than HILP across the tested chaser speeds. This is task transfer under unchanged dynamics, not adaptation to changed dynamics or observations.

### Layout and control generalization

- In **Diverse PointMaze**, models train on 1M random-policy transitions from 5, 10, 20, or 40 layouts and are evaluated on disjoint layouts. All methods obtain 97–100% on a single fixed maze, but only PLDM remains consistently successful as held-out layouts become more different from the training layouts. The layout is visible in the input, so this result shows conditional-dynamics/spatial generalization—not recovery from a purely visual domain shift.
- In state-based **Ant U-Maze** (29-dimensional state, 8-dimensional action), PLDM, HILP, and HIQL achieve 100% success in the short-trajectory stitching settings where several other baselines fail. Evaluation uses only ten trials per method per setting, so this is promising rather than broad evidence for high-dimensional robotics.

### Objective and planner ablations

The full PLDM obtains **98.0 ± 1.5%** in Two-Rooms and **98.7 ± 2.8%** in Diverse PointMaze. Removing objective terms gives:

| Removed term | Two-Rooms success | Diverse PointMaze success |
|---|---:|---:|
| variance | 13.4 ± 9.2 | 11.4 ± 6.5 |
| covariance | 29.2 ± 4.4 | 7.8 ± 4.1 |
| temporal smoothness | 71.0 ± 3.0 | 95.6 ± 3.2 |
| IDM | 98.0 ± 1.5 | 75.5 ± 8.2 |

Thus anti-collapse regularization is essential in these implementations; temporal smoothness and IDM are environment-dependent rather than universally necessary.

On high-quality Two-Rooms data, PLDM reaches **97.4 ± 1.3%**, compared with DreamerV3 reconstruction at 24.0 ± 6.9%, a same-architecture pixel-reconstruction model at 26.2 ± 13.9%, reward-free TD-MPC2 at 0%, and TD-MPC2 + IDM at 35% (the last result has one seed). The authors explicitly call the modified Dreamer comparison flawed because Dreamer's discrete representation was not designed for their latent-distance planner. These results support PLDM's chosen objective in this testbed, not a general theorem that reconstruction is unsuitable.

Replanning every step costs **16.0 s per 200-step Two-Rooms episode**, versus 3.6 s for GCIQL and 4.0 s for HIQL. Replanning every 4 steps lowers PLDM to 4.8 s while retaining 95% of its every-step success; every 16 steps takes 2.6 s and retains 90%.

## What is relevant for WhiteHole

1. **Downstream planning is the claim-bearing metric.** WhiteHole already measures linear-probe quality, paired latent error, and action-conditioned rollout error in [`scripts/eval_jepa_baseline.py`](../../../scripts/eval_jepa_baseline.py) and [`whitehole/adaptation/adapters.py`](../../../whitehole/adaptation/adapters.py). These are useful diagnostics, but PLDM's strongest evidence comes from closed-loop MPPI success. WhiteHole's existing [`WallMPCEvaluator`](../../../whitehole/planning/wall/mpc.py) reports goal error and wall-crossing behavior and should be the final evaluator for an adapted model.
2. **The anti-collapse ablations motivate safeguards, not direct transplantation.** Variance/covariance preservation and IDM are plausible adapter regularizers, but the PLDM results concern end-to-end source-model training. WhiteHole must test whether they improve a frozen model's planning rather than assuming the ablation transfers.
3. **Data stress tests are reusable.** Short sequences, random actions, missing doorway crossings, dataset-size sweeps, and held-out layouts isolate different failure modes. The official [Two-Rooms environment](https://github.com/vladisai/PLDM/tree/1bd7e564ecd961205bc18b23067b19e9ca24ac90/pldm_envs/wall) and [Diverse PointMaze generator](https://github.com/vladisai/PLDM/tree/1bd7e564ecd961205bc18b23067b19e9ca24ac90/pldm_envs/diverse_maze) are useful reference/environment sources under the repository's [MIT license](https://github.com/vladisai/PLDM/blob/1bd7e564ecd961205bc18b23067b19e9ca24ac90/LICENSE).
4. **Do not conflate layout generalization with visual adaptation.** PLDM trains across rendered layouts and allows the encoder/predictor to change. WhiteHole preserves dynamics and action semantics while changing rendering, then freezes the source world model. A successful WhiteHole adapter must restore planner-relevant coordinates under that stricter intervention.

**Evidence rule for WhiteHole:** a lower latent MSE, a better position probe, or even an open-loop rollout improvement is not sufficient to claim that a world model was adapted. Such a claim requires statistically repeated **closed-loop downstream planning recovery** under the target observation shift, with the same planner and compute budget, relative to source-oracle, shifted-unadapted, and appropriate upper-bound conditions.

## Concrete experiments to run next

- **Planning-gated appearance benchmark.** For mild, medium, and hard Two-Rooms shifts, compare (i) source-render oracle, (ii) shifted observations with no adapter, (iii) each trained adapter, and (iv) a target-domain end-to-end model as an upper bound. Freeze the source encoder and predictor in (ii–iii), use identical MPPI samples/horizon/replanning frequency, and report success, final goal error, wall-crossing rate, steps-to-goal, and runtime over at least three seeds. Report probe and rollout metrics only alongside these planning outcomes.
- **Adapter-objective ablation.** At fixed adapter capacity and target-transition budget, ablate one-step dynamics alignment, multi-step rollout alignment, paired latent alignment, local-isometry/identity terms, and variance/covariance matching. Evaluate ConstantOffset, DiagonalAffine, and one higher-capacity residual adapter. A component counts as useful only if it improves held-out target-domain planning, not merely its own training loss.
- **Target-data quality and quantity sweep.** Train the same frozen-model adapter with approximately 634, 1.3K, 5K, 20K, and 80K target transitions, separately using random local motion and doorway-crossing trajectories. Test long-horizon cross-room planning. This distinguishes visual coordinate recovery from unsupported dynamics stitching and checks whether adapter sample efficiency follows PLDM's source-training result.
- **Horizon/compounding study.** Train adapters with rollout horizons 1, 4, 8, and 15, then evaluate open-loop error by horizon and closed-loop planning with short and long MPPI horizons (for example 16, 32, and 96). Require the selected horizon to improve target planning across seeds; rollout-vs-persistence ratios alone cannot establish world-model utility.
- **Appearance × layout factorial.** Cross source/target appearance with seen/held-out wall and doorway layouts while keeping collision rules and action semantics fixed. Train the adapter only on source layouts, then evaluate all four cells with source-oracle and unadapted controls. This tests whether an appearance adapter transfers compositionally without mislabeling physical-layout generalization as domain adaptation.

## Risks / open questions

- **Narrow setting:** the paper assumes deterministic, fully observed dynamics and evaluates navigation only. It does not establish results for partial observability, stochastic transitions, manipulation, or natural-image shifts.
- **Coverage remains fundamental:** PLDM can compose short observed transitions but performs poorly when the doorway transition is absent. Planning cannot validate unsupported dynamics merely by producing a plausible latent path.
- **Latent distance is task-sensitive:** Euclidean closeness to an encoded goal is not guaranteed to represent reachability. Collapse prevention and action sensitivity are necessary, and closed-loop evaluation is still required.
- **Reproducibility mismatch:** the v4 appendix reports 500 MPPI samples and ensemble uncertainty regularization, while the inspected public Two-Rooms YAML uses [2,000 samples and noise 12](https://github.com/vladisai/PLDM/blob/1bd7e564ecd961205bc18b23067b19e9ca24ac90/pldm/configs/wall/icml/seqlen17_3M.yaml), and the inspected `JEPA`/planner paths expose a single predictor without an obvious paper-specific ensemble-disagreement path. Reproductions should record exact paper-versus-code choices rather than treating the repository defaults as canonical.
- **Baseline asymmetry:** methods receive substantial dataset-specific tuning. The Dreamer modification is acknowledged as architecturally mismatched, and TD-MPC2 + IDM has one seed. These comparisons should not be reused as blanket claims about reconstruction or model-free RL.
- **Synthetic pairing can overstate adaptation:** WhiteHole can generate paired source/shifted frames exactly, unlike most real shifts. Results should separate paired supervision from transition-only adaptation and preserve the same downstream planning gate for both.
