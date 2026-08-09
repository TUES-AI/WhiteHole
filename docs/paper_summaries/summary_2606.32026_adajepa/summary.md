# AdaJEPA: An Adaptive Latent World Model

**Paper:** Ying Wang, Oumayma Bounou, Yann LeCun, and Mengye Ren, [arXiv:2606.32026v1](https://arxiv.org/abs/2606.32026v1), submitted 2026-06-30.  
**Project/code:** [project page](https://agenticlearning.ai/adajepa/) · [official repository](https://github.com/agentic-learning-ai-lab/adajepa) · inspected commit [`a29975964f966f2836a2c7e26f464367c795c333`](https://github.com/agentic-learning-ai-lab/adajepa/tree/a29975964f966f2836a2c7e26f464367c795c333) (2026-07-19).  
**Primary material read:** the complete v1 TeX tree (`main.tex`, every section input, bibliography, equations, algorithm, tables, appendices, and figure contents) and the released adaptation/MPC implementation. The quantitative claims below are author-reported and have not been independently reproduced in WhiteHole.

## Problem and core idea

A frozen latent world model can make MPC optimize actions against the wrong imagined dynamics, particularly after a visual, physical, or layout shift. AdaJEPA turns each action executed by MPC into a test-time training sample. Within each episode it repeatedly:

1. plans with the current encoder and predictor;
2. executes the first action chunk;
3. stores the observed transition;
4. takes usually one gradient step on latent prediction error; and
5. replans with the updated model.

The important contribution is therefore the **coupling of online self-supervision to receding-horizon control**, not a new JEPA architecture. No reward or expert label is required, but real online interaction is required. Adaptation is episode-local: every evaluation episode starts from the pretrained weights, maintains its own buffer/model copy, and discards the updates afterward. The evidence does not establish continual adaptation across episodes or long-term resistance to forgetting.

For WhiteHole, this is a strong alternative to training an adapter offline before deployment. It does **not** directly validate WhiteHole's stricter frozen-world-model objective: the paper normally changes both the source predictor and part of the source encoder, whereas WhiteHole studies small input/latent adapters with the source encoder and predictor frozen.

## Method details (short)

### World model and source training

The model has a sensory encoder \(\mathcal E^s_\phi\), action encoder \(\mathcal E^a_\psi\), and latent predictor \(f_\theta\):

\[
z_t=\mathcal E^s_\phi(o_t),\qquad u_t=\mathcal E^a_\psi(a_t),\qquad
\hat z_{t+1}=f_\theta(z_t,u_t).
\]

The main models use a small ResNet visual encoder with global features and a ViT-style transformer predictor. Visual, proprioceptive, and action embeddings are concatenated before prediction. Training uses three history frames, frame skip 5, one-step latent MSE with a stop-gradient target, and temporal-curvature regularization inherited from Temporal Straightening. The paper also evaluates spatial-feature and DINO-WM variants. The action encoder is trainable during source training (learning rate \(5\times10^{-4}\)) but is **not selected for online updates** in the reported default or released adaptation code.

### Online loss and replay

For online buffer \(\mathcal B\), AdaJEPA minimizes

\[
\mathcal L_{\rm ada}(\mathcal B)=\frac{1}{|\mathcal B|}
\sum_{(o_i,a_i,o_{i+1})\in\mathcal B}
\ell\!\left(f_\theta(\mathcal E^s_\phi(o_i),\mathcal E^a_\psi(a_i)),
\operatorname{sg}(\mathcal E^s_\phi(o_{i+1}))\right).
\]

The implementation applies MSE to visual/proprioceptive latent dimensions, excludes action dimensions, and averages over valid sliding history windows ([loss implementation](https://github.com/agentic-learning-ai-lab/adajepa/blob/a29975964f966f2836a2c7e26f464367c795c333/planning/adajepa.py#L208-L235)). With encoder adaptation enabled, prediction-input encodings receive gradients while next-observation target encodings are detached when the pretrained model uses stop-gradient; targets are re-encoded before each additional update. The paper says removing stop-gradient gives similar planning when only the last encoder/predictor layers are updated for one step, but provides no numeric table for that claim.

The default buffer is `recent5`: five latest executed chunks. `hard-N` instead retains chunks with the highest current prediction loss; no buffer and an effectively infinite buffer are controls. The released implementation merges contiguous recent chunks and keeps hard-buffer chunks separate ([MPC integration](https://github.com/agentic-learning-ai-lab/adajepa/blob/a29975964f966f2836a2c7e26f464367c795c333/planning/adajepa_mpc.py#L127-L189)). Replay choice is secondary in the PushObj ablation: success ranges 81–87% for seen T and 35–44% for unseen square, versus frozen 50% and 20%, respectively.

### Which parameters are updated

The default takes one Adam step per replan with the same rates used in source training:

- predictor: \(5\times10^{-4}\), last transformer block plus final LayerNorm;
- encoder: \(10^{-5}\), final projection head (`Linear–GELU–Linear–LayerNorm`);
- all other encoder/predictor parameters and the action encoder: frozen.

Reported direct-update variants are `predlast+enclast`, `predfirst+enclast`, `predfirstlast+enclast`, `predlast+encfirst` (ResNet `rb1`, 3→32 channels), and predictor-only `predlast+encfrozen`. The predictor's first block is especially effective for held-out maze layouts, consistent with a mismatch near latent/action inputs rather than only at the output.

The paper's LoRA variant inserts rank-8, \(\alpha=16\) adapters into **every linear layer of the full encoder and predictor**, updating only LoRA parameters. It usually helps but does not dominate selected-layer finetuning. The released repository at the inspected commit contains no LoRA implementation, and neither the paper nor code reports trainable parameter counts; “lightweight” therefore cannot be checked as an equal-parameter comparison.

The released parameter selection also has architecture-specific behavior: for pretrained visual backbones it updates non-backbone projector parameters, while for a scratch encoder it selects the last child module ([selection code](https://github.com/agentic-learning-ai-lab/adajepa/blob/a29975964f966f2836a2c7e26f464367c795c333/planning/adajepa.py#L116-L171)). Thus “encoder last layer” is not one invariant intervention across model families.

### Coupling to MPC

The paper writes a generic weighted latent goal cost

\[
\arg\min_{a_{t:t+H-1}}\sum_{k=1}^{H}\alpha_k
\|\hat z_{t+k}-z_g\|_2^2,
\qquad z_g=\mathcal E^s_\phi(o_g).
\]

In the released PushObj configs, a nominal 25-environment-step horizon is converted by frame skip 5 into five model action chunks. MPC executes five raw actions (one model chunk), observes the resulting transition, adapts, and replans; the maximum is normally 20 replans. The main PushObj curves extend this to 30 replans. Action optimization uses either:

- gradient planning: Adam, zero action initialization, learning rate 0.1, 100 steps; or
- CEM: 200 samples, 30 elites, 10 iterations.

Unexecuted actions warm-start the next plan. PushObj uses a staged terminal-then-weighted-full-horizon cost, while PointMaze uses all rollout states; these released objectives are more specific than the generic paper equation ([objective code](https://github.com/agentic-learning-ai-lab/adajepa/blob/a29975964f966f2836a2c7e26f464367c795c333/planning/objectives.py), [PushObj GD config](https://github.com/agentic-learning-ai-lab/adajepa/blob/a29975964f966f2836a2c7e26f464367c795c333/conf/adajepa_plan_gd_pushobj.yaml)). After encoder updates, current and goal observations are both re-encoded for the next plan, so the planning metric itself moves along with the representation.

## Key results

### Parameter-placement ablation

The appendix reports environment-averaged success at the normal 20-step budget. Values below are percentages; no uncertainty is shown for this summary figure.

| Shift / planner | Frozen | predlast + enclast | predfirst + enclast | Best direct update | LoRA |
|---|---:|---:|---:|---:|---:|
| Shape / GD | 42 | 68 | 68 | 68 | 68 |
| Shape / CEM | 39 | 65 | 67 | 67 | 65 |
| Visual / GD | 50 | 60 | 61 | 64 | 62 |
| Visual / CEM | 53 | 62 | 66 | 67 | 65 |
| Dynamics / GD | 79 | 80 | 82 | 82 | 80 |
| Dynamics / CEM | 81 | 83 | 83 | 85 (`predlast+encfirst`) | 82 |
| Layout / GD | 53 | 66 | **79** | **79** | 66 |
| Layout / CEM | 49 | 55 | **71** | **71** | 57 |

This supports an environment-dependent placement hypothesis:

- shape changes can be corrected largely in the predictor; freezing the encoder is competitive;
- visual shifts benefit from encoder updates;
- held-out connectivity benefits strongly from the predictor's first block;
- LoRA is viable but not consistently superior.

However, the paper's statement that **all** choices improve is not literally true in every plotted cell: predictor-only adaptation gives 48% on layout/CEM versus 49% frozen. Aggregated bars also hide individual shift conditions and show no confidence intervals.

### Dynamics and layout

The more detailed PointMaze table reports:

- **Default dynamics:** frozen 82.7±6.8 (GD), 84.0±3.3 (CEM); default adaptation 83.3±6.6 and 83.3±3.4.
- **Low mass ×0.2:** frozen 77.3±8.2 / 82.0±2.8; default adaptation 80.0±3.3 / 86.7±2.5.
- **High damping ×20:** frozen 77.3±5.0 / 76.0±2.8; default adaptation 77.3±10.5 / 78.7±3.4.
- **Held-out layouts:** frozen 53.3±8.2 / 49.3±6.2; `predlast+enclast` 66.0±7.1 / 55.3±5.0; `predfirst+enclast` **78.7±5.0 / 70.7±3.8**.

The layout gains are large. Dynamics gains are mostly modest because the frozen history-conditioned model is already robust. The claim that adaptation is “safe” in-distribution is too strong: default-dynamics CEM drops 0.7 points under the default update, albeit well within the reported variation.

### Model variants, latency, and data scale

On in-distribution PushT validation trajectories:

| Base world model | GD frozen → adapt | CEM frozen → adapt | Added reported time/replan |
|---|---:|---:|---:|
| Temporal Straightening, global 1×384 | 84.0 → 85.3 | 74.0 → 81.3 | 0.03 s |
| Temporal Straightening, spatial 196×384 | 91.3 → 92.0 | 89.3 → 93.3 | 0.01–0.02 s |
| DINO-WM, spatial 196×384 | 68.0 → 70.0 | 86.7 → 90.0 | 0.02–0.03 s |

Timing uses one H200 and is reported per replan. The official adaptive planner processes episodes independently/sequentially to isolate their weights, while the README says the frozen baseline plans the batch together. Consequently, the small per-sample update overhead does not establish equal batched throughput or deployment latency.

The strongest sample-scale result is on PushObj. With one training shape and 1k trajectories, seen-shape success rises from 28.1% to 60.8%; that adapted model beats a frozen one trained on the same shape with 16k trajectories (43.5%). Across the full scale grid, adaptation improves every shown cell. Diversity still matters: at 16k total trajectories, four shapes ×4k gives 51.9% adapted success on unseen shapes, versus 45.8% for one shape ×16k. This is evidence that online specialization complements coverage, not that it replaces diverse source data.

## What is relevant for WhiteHole

1. **Online transition-only adaptation is a directly relevant control.** WhiteHole already computes transition and multistep consistency for post-encoder adapters in [`whitehole/adaptation/adapters.py`](../../../whitehole/adaptation/adapters.py). AdaJEPA suggests moving one optimization step into the MPC feedback loop and resetting per episode, using only transitions that the planner actually causes.
2. **Placement should be diagnosed by shift mechanism, not chosen globally.** WhiteHole has pre-encoder affine/residual image adapters in [`scripts/train_input_film_adapter.py`](../../../scripts/train_input_film_adapter.py) and target-latent→source-latent adapters before the frozen predictor. AdaJEPA's visual/predictor/layout split motivates an equal-budget comparison among input adapter, latent adapter, encoder head, predictor input block, and predictor output block.
3. **AdaJEPA relaxes WhiteHole's frozen-model premise.** Updating predictor/encoder weights can repair the model but no longer demonstrates that a fixed source world model is reusable. WhiteHole should report this as an online-finetuning upper bound or separate method, not conflate it with frozen-model adaptation.
4. **The objective is not source-coordinate alignment.** AdaJEPA reduces self-consistency in a representation that may move, while WhiteHole wants target observations mapped into coordinates compatible with a frozen predictor. A lower online MSE can result from co-adapting encoder and predictor without recovering the source latent geometry; source-oracle rollout and downstream MPC remain necessary.
5. **MPC is both evaluator and data collector.** WhiteHole's current MPC loop replans but does not update the model. Coupling adaptation to its feedback changes the visited-state distribution, so comparisons must fix planner rollouts/action budget and separate better modeling from policy-induced data selection.

## Concrete experiments to run next

**These are WhiteHole proposals, not paper results. Each includes a falsifiable placement or objective hypothesis.**

1. **Shift-mechanism × update-placement factorial.** On source, mild/medium/hard Two-Room visual shifts, and one controlled dynamics shift, compare frozen, input affine/residual adapter, post-encoder latent adapter, encoder head, predictor first block, predictor last block, and rank-8 LoRA at matched trainable-parameter and online-transition budgets. Evaluate source-oracle rollout error and closed-loop MPC success over multiple model/evaluation seeds. Hypothesis: visual shifts favor input/encoder placement and dynamics shifts favor predictor placement; it is falsified if predictor-only consistently wins visual shifts or input-only wins dynamics shifts.
2. **One-step self-prediction versus source-compatible objectives.** At one fixed placement, compare AdaJEPA one-step MSE with stop-gradient, no stop-gradient, WhiteHole's multistep self-rollout loss, source-rollout alignment, and paired source-latent alignment as a synthetic oracle. Track per-horizon rollout-to-source error, persistence ratio, latent rank/variance, and planning. The “same JEPA loss is sufficient” hypothesis is falsified if online MSE falls while source rollout or planning stagnates/collapses, or if multistep/source-aligned losses reliably recover planning at equal updates.
3. **Causal MPC-coupling control.** Give four conditions exactly the same collected transitions and SGD steps: immediate adapt-before-next-replan, one-replan-delayed adaptation, adaptation only after the episode, and shuffled-action/next-state transitions. Hold action proposals fixed where possible. The plan–adapt coupling hypothesis is falsified if delayed or post-episode updates match immediate online planning, or shuffled transitions improve similarly; that would indicate generic specialization/regularization rather than calibrated action-conditioned dynamics.
4. **Moving-coordinate/goal test.** For encoder updates, compare re-encoding the goal after every update (AdaJEPA), freezing the original goal latent, adding source-coordinate anchoring, and predictor-only adaptation. Measure current-goal latent geometry and planner action changes after each update. The hypothesis that encoder adaptation corrects perception rather than merely co-warps the planning metric is falsified if gains disappear with a fixed goal latent and source-coordinate anchoring, while self-prediction loss still improves.
5. **Direct layers versus LoRA under equal capacity.** Match trainable parameters across selected-block direct updates and LoRA ranks/placements (predictor-only, encoder-only, both), and sweep predictor first versus last block. The hypothesis that last-layer direct updates are a robust lightweight default is falsified if equal-budget input-block LoRA/direct updates consistently dominate across visual and layout shifts, or if apparent LoRA underperformance vanishes after parameter/optimizer matching.
6. **Replay locality under a changing shift.** Within one episode, switch appearance or dynamics halfway through and compare no buffer, `recent1`, `recent5`, `hard5`, and infinite replay. Report adaptation lag, post-switch planning, and pre-switch forgetting. The local-calibration hypothesis is falsified if recent replay offers no faster recovery than infinite/hard replay, or if hard-example replay improves loss while degrading MPC.

## Risks / open questions

- **Very recent, single-version preprint:** this is arXiv v1. The official code was released about three weeks later and has not been independently validated here. The repository contains no LoRA implementation for a central placement ablation.
- **Limited statistical evidence:** main results average three test-data seeds with 50 episodes each, but the paper does not establish variation over independently trained world-model seeds or define every shaded/± statistic. Several broad claims rest on small changes relative to that variation.
- **No strong online-adaptation baseline:** comparisons are primarily frozen versus the same model with prediction-loss updates. There is no matched adaptive-control/system-identification baseline, entropy/consistency TTA baseline, or objective ablation showing that latent prediction MSE is the uniquely useful signal.
- **Narrow shifts and tasks:** all tests are simulator goal reaching. Visual corruptions/colors, object geometry, MuJoCo mass/damping, and held-out mazes are controlled; there is no real camera drift, stochastic dynamics, partial observability, safety constraint, manipulation robot, or adversarial/nonstationary deployment.
- **Evaluation construction can favor recoverability:** PushObj test trajectories are filtered to contain contact, goals are often sampled 25 steps away, and layout goals are constrained to feasible 3–5-cell paths. These are useful controls but not broad deployment evidence.
- **Interaction and reset assumptions matter:** adaptation consumes on-policy transitions, incurs real environment actions before receiving supervision, and starts over from pretrained weights every episode. It cannot correct the first action chunk and does not test accumulated forgetting or transfer between episodes.
- **Moving latent targets can hide failure:** re-encoding the goal after encoder updates keeps planning internally consistent but makes lower self-prediction loss insufficient evidence of source-coordinate recovery. Decoder visualizations staying near the training manifold are qualitative and do not resolve this ambiguity.
- **Protocol details vary by figure:** the default maximum is 20 replans, while the principal PushObj curves use 30 to show continued gains. Per-replan H200 timing does not measure batched throughput, total online interaction cost, or deployment hardware latency.
