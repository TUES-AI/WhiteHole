# A Path Towards Autonomous Machine Intelligence

**Yann LeCun — version 0.9.2, 2022-06-27**  
**Primary record:** [OpenReview BZ5a1r-kVsf](https://openreview.net/forum?id=BZ5a1r-kVsf) ([PDF endpoint](https://openreview.net/pdf?id=BZ5a1r-kVsf))

> **Source limitation.** This paper has no arXiv record or public TeX source. The official OpenReview pages were access-blocked during retrieval, so this summary could not be prepared from an official source package. It is based on the complete text extracted at `/tmp/whitehole-paper-sources/lecun-path/paper.txt` from a PDF mirror whose title, author, version (0.9.2), and date (2022-06-27) match the official record. Consequently, the manuscript content was read in full, but OpenReview metadata, revision history, and exact PDF provenance could not be independently verified here.

## Problem and core idea

The paper asks how an agent could learn world knowledge mostly from observation, reason and plan through trainable machinery, and represent states and actions at several abstractions and time scales. Its answer is a proposed cognitive architecture centered on a **configurable predictive world model**. The agent predicts in learned representation space, evaluates imagined trajectories with intrinsic and learned costs, and either acts reactively or optimizes an action sequence through the model.

The centerpiece is the **Joint-Embedding Predictive Architecture (JEPA)**: encode observed/context input `x` and target/future input `y` as `s_x` and `s_y`, then predict `s_y` from `s_x`, optionally using a latent variable `z`. Prediction in representation space is intended to discard hard-to-predict, task-irrelevant pixel details while retaining informative and predictable state. A proposed **Hierarchical JEPA (H-JEPA)** stacks such models so detailed representations support short-horizon prediction and coarser representations support longer-horizon prediction and planning.

This is explicitly a **position paper**, not a conventional technical or experimental paper (Prologue). It assembles a research agenda and identifies missing mechanisms; it does not demonstrate that the complete architecture works.

## Method details (short)

### Agent architecture

The proposal contains six interacting modules (Sections 3–6; Figure 2):

- **Perception** encodes sensor input into a hierarchical state representation, filtered for the current task.
- **World model** completes missing state and predicts future representations conditioned on candidate actions; latent variables represent information not predictable from context.
- **Cost** sums immutable intrinsic-cost terms and trainable critic terms. The critic predicts future intrinsic cost.
- **Actor** either emits a reactive policy action (**Mode-1**) or searches for a low-cost action sequence through the world model (**Mode-2**, receding-horizon MPC).
- **Short-term associative memory** stores states and costs and supports critic training and state tracking.
- **Configurator** modulates the other modules for a task, e.g. through attention, routing, parameters, cost weights, or conditioning tokens.

The paper proposes distilling expensive Mode-2 plans into Mode-1 policies by training a policy to imitate optimized actions (Section 3.1.3; Figure 5).

### JEPA objective and collapse prevention

For encoders `s_x = g_x(x)` and `s_y = g_y(y)`, the latent-variable JEPA energy is

`E(x, y, z) = D(s_y, Pred(s_x, z))`, with `F(x, y) = min_z E(x, y, z)`.

Multiple compatible futures can be represented either by invariances in the target encoder or by varying `z` (Section 4.4; Figure 12). The proposed non-sample-contrastive training principle combines four requirements (Section 4.5; Figure 13):

1. keep `s_x` informative about `x`;
2. keep `s_y` informative about `y`;
3. make `s_y` predictable from `s_x`;
4. minimize the information carried by `z` so the predictor cannot ignore context and copy the target through the latent.

The paper points to [VICReg](https://arxiv.org/abs/2105.04906) as one realization of the first three requirements: variance terms prevent constant dimensions, covariance terms reduce redundancy, and an invariance/prediction term aligns representations. Proposed controls on `z` include low dimension, discreteness, sparsity, and noise. These are design candidates, not compared solutions.

### Hierarchy, planning, and uncertainty

H-JEPA is proposed as a temporal hierarchy: lower levels retain detail and predict short horizons, while upper levels temporally pool lower-level representations and predict farther ahead (Section 4.6; Figure 15). For planning, a high-level action is treated as a condition or subgoal for the level below; high- and low-level action variables may be optimized jointly (Section 4.7; Figure 16).

Under uncertainty, each predictor can use regularized latent variables. Planning samples or searches latent configurations, rolls out several possible state trajectories, and selects actions by expected cost or a risk-sensitive combination of cost mean and variance (Section 4.8; Figure 17). The paper acknowledges exponential trajectory growth and suggests pruning, beam-like search, or MCTS rather than providing a solved inference procedure.

## Key results

There are **no experimental results**: no implementation of the assembled architecture, datasets, quantitative tables, benchmark comparisons, ablations, or statistical tests are reported. The paper's “contributions” are architectural and conceptual proposals.

The main conclusions should therefore be read as hypotheses:

- latent prediction may avoid spending capacity on unpredictable pixel details;
- regularized joint embeddings may learn representations that are both informative and predictable without sample negatives;
- stacking predictive representations may produce useful temporal abstractions;
- differentiable latent dynamics and costs may enable planning by gradient optimization;
- regularized latent variables may represent multiple plausible futures.

The manuscript relates these claims to prior work on VICReg/Barlow Twins, energy-based models, learned dynamics, MPC, memory networks, and hierarchical control, but evidence for those ingredients does **not** validate their proposed integration. Section 8.1 explicitly leaves central questions unresolved: whether H-JEPA can learn an abstraction hierarchy from video, how best to regularize uncertainty latents, how to search actions and latent futures, and how the configurator should discover subgoals.

## What is relevant for WhiteHole

WhiteHole studies **frozen latent world models under visual observation shifts**, whereas this paper addresses how to learn and use a full autonomous agent. It does not propose or evaluate domain adaptation. Its relevance is therefore indirect but foundational:

- **Predictability is a useful adaptation constraint.** A target observation adapter should map shifted inputs or latents into coordinates where the frozen source predictor remains accurate, rather than merely matching marginal latent statistics. WhiteHole's transition-alignment and multistep-rollout losses instantiate this principle.
- **Informative representations must accompany predictable ones.** Dynamics consistency alone can admit collapsed or low-information adapter outputs. VICReg-style per-dimension variance/covariance diagnostics or losses provide a paper-motivated complement, although preserving source geometry is not identical to maximizing information.
- **Adapt only nuisance variation.** JEPA's intended invariance to unpredictable, irrelevant appearance suggests evaluating whether an adapter removes visual style changes while preserving position, velocity, object identity, and action-conditioned dynamics.
- **Long horizons expose false alignment.** One-step agreement can hide compounding coordinate errors. The hierarchy argument motivates reporting adaptation quality by rollout horizon and, eventually, by temporal scale rather than relying on a single latent MSE.
- **Planning is the consequential test.** Reduced latent prediction error matters only if the adapted frozen model still supports goal-conditioned MPC. WhiteHole should keep representation, rollout, probe, and closed-loop planning metrics separate.
- **Task/shift conditioning is analogous to configuration.** A small conditioned input or latent adapter can test reuse of one frozen world model across visual shifts without claiming to implement the paper's much broader configurator.

The paper does **not** imply that a JEPA is naturally invariant to WhiteHole's shifts, that a frozen predictor uniquely identifies the correct source coordinates, or that improved self-supervised loss necessarily improves control. Those are empirical questions for this repository.

## Concrete experiments to run next

1. **Predictability versus anti-collapse ablation on Two Rooms.** With the encoder and predictor frozen, train the same adapter family using (a) one-/multistep dynamics consistency only, (b) dynamics plus target variance/covariance regularization, and (c) dynamics plus source variance/covariance matching. Report per-dimension standard deviation, effective rank, one-step error, horizon-wise rollout error, location-probe error, and MPC success. This tests whether JEPA-style information preservation prevents dynamics-compatible collapse.

2. **Horizon curriculum for adaptation.** Train otherwise identical target-to-source adapters with rollout horizons `{1, 4, 8, 15}` and a mixed-horizon objective. Evaluate all models at every horizon plus closed-loop planning under the same mild/medium/strong appearance shifts. This directly tests the paper's claim that predictability at multiple time scales matters, without requiring a new hierarchy.

3. **Paired supervision as an oracle, not the proposal.** On synthetic Two Rooms shifts, compare unpaired dynamics-only adaptation against direct paired source/target latent alignment and their combination. Use the paired map only as an oracle diagnostic, then measure how closely unpaired objectives recover source latent neighborhoods and downstream MPC. This identifies non-identifiability that prediction consistency alone may conceal.

4. **Shift-configured versus separate adapters.** Compare one small FiLM or residual adapter conditioned on shift identity/severity with independently trained adapters and one unconditioned shared adapter. Freeze the same world model and balance samples across shifts. Evaluate held-out shift strengths as well as seen shifts to test the limited WhiteHole analogue of a configurable, reusable world model.

5. **Uncertainty-aware adaptation stress test.** Add controlled partial observability or stochastic visual distractors, then compare a deterministic adapter with a low-dimensional stochastic/ensemble adapter. Measure best-of-`K` and mean rollout error, trajectory coverage/calibration, and expected-cost versus variance-penalized MPC. This tests whether extra latent capacity represents genuine ambiguity or merely bypasses the frozen dynamics; latent dimension and information regularization must be ablated.

## Risks / open questions

- **No empirical support in this paper:** all WhiteHole mappings above are deductions from a position paper, not reproduced findings.
- **Dynamics-preserving maps are not unique:** an adapter may satisfy the frozen predictor while rotating, collapsing, or otherwise distorting state information needed by probes and planners.
- **Variance/covariance is only a proxy for information:** VICReg-style moments can prevent simple collapse but do not guarantee semantics, invertibility, or source-coordinate recovery.
- **Hierarchy is underspecified:** the manuscript gives architectural diagrams but no validated learning schedule, temporal abstraction criterion, or subgoal-discovery algorithm.
- **Latent uncertainty can become a shortcut:** an over-capacity `z` can carry target information and flatten the energy; tight capacity controls and calibration tests are essential.
- **Frozen-model limits remain:** if an observation shift removes task-relevant information or exposes source-model errors, no lightweight adapter can recover a valid source state.
- **Planning can exploit model error:** lower latent losses do not guarantee safer or better trajectories, especially over long horizons or under distribution shift.
