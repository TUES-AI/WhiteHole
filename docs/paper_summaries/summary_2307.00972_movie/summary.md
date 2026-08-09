# MoVie: Visual Model-Based Policy Adaptation for View Generalization

**Paper:** Sizhe Yang, Yanjie Ze, and Huazhe Xu, NeurIPS 2023, arXiv:2307.00972v3 — [arXiv](https://arxiv.org/abs/2307.00972v3) · [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/43b77cef2a83a25aa27d3271d209e4fd-Abstract-Conference.html) · [project page](https://yangsizhe.github.io/MoVie/) · [official code](https://github.com/yangsizhe/MoVie/tree/f721518846ffd1de4fdf09a4fc380b720ad04183)

This summary is based on the complete supplied TeX source, including all tables and the appendix, and cross-checked against official code commit `f721518846ffd1de4fdf09a4fc380b720ad04183`. All quantitative results are **paper claims, not WhiteHole results**.

## Problem and core idea

A pixel-based controller trained from one fixed camera can fail when the deployment camera changes even though the physical state and transition dynamics are unchanged. MoVie treats this as a test-time **perception-alignment** problem rather than retraining the policy to be view invariant.

The source TD-MPC or MoDem agent is trained normally. At deployment, MoVie collects reward-free online transitions `(observation, action, next observation)`, keeps the learned latent dynamics and control machinery fixed, and adapts only a spatially augmented image encoder. The frozen source dynamics acts as self-supervision: target-view observations should be encoded into the source latent coordinate system in which the old action-conditioned dynamics—and therefore the old controller—remain valid.

The paper evaluates four camera shifts:

1. **Novel view:** one fixed camera displacement, with orientation continuing to face the agent or a fixed scene point.
2. **Moving view:** the camera follows a predefined back-and-forth trajectory while looking at that target.
3. **Shaking view:** fresh Gaussian position noise is applied every time step. The appendix specifies `σ=0.04 m` for DMControl/Adroit and `σ=0.4 m` for xArm, clipped to `[-0.07,0.07] m` in all cases.
4. **Novel FOV:** a one-time field-of-view increase: `45°→53°` for DMControl, `50°→60°` for xArm, and `45°→50°` for Adroit. A small-FOV Cheetah ablation is also reported.

## Method details (short)

Let `h` be the image encoder, `d` the action-conditioned latent dynamics, and `π` the source policy/control system. Source training includes the latent transition objective

\[
\mathcal L_{\text{dynamics}}
=\left\|d(h(o_t),a_t)-h(o_{t+1})\right\|_2.
\]

At test time, the source dynamics becomes fixed supervision, `d*`, and the encoder becomes a spatial adaptive encoder (SAE):

\[
\mathcal L_{\text{view}}
=\left\|d^*(h^{\mathrm{SAE}}(o_t),a_t)-h^{\mathrm{SAE}}(o_{t+1})\right\|_2.
\]

Both encoder occurrences are trainable; the paper does not describe a detached target encoder. The [released implementation](https://github.com/yangsizhe/MoVie/blob/f721518846ffd1de4fdf09a4fc380b720ad04183/src/algorithms/tdmpc/src/algorithm/tdmpc.py#L288-L343) uses MSE for this one-step loss.

### What changes and what stays frozen

- **Adapted:** two identity-initialized spatial transformer networks (STNs), plus the original visual encoder at a much smaller learning rate.
- **Frozen/unchanged:** latent dynamics, reward and value models, policy, planning procedure, and all other source-agent components. In released TD-MPC code the dynamics is excluded from adaptation optimizers rather than globally marked `requires_grad=False`.
- **Placement:** one STN is inserted before the first convolution and one after it ([SAE construction](https://github.com/yangsizhe/MoVie/blob/f721518846ffd1de4fdf09a4fc380b720ad04183/src/algorithms/tdmpc/src/algorithm/tdmpc.py#L67-L95)). Each STN predicts six parameters for a 2D affine warp. The input STN transforms every RGB frame in a frame stack independently, which is intended to handle camera motion between frames ([implementation](https://github.com/yangsizhe/MoVie/blob/f721518846ffd1de4fdf09a4fc380b720ad04183/src/algorithms/tdmpc/src/algorithm/transform.py#L8-L36)).
- **Online data use:** appendix defaults are replay capacity 256, batch size 32, and 32 updates per interaction. STN learning rate is `1e-5`; encoder learning rate is `1e-6` on xArm and `1e-7` otherwise. Twenty consecutive episodes are evaluated, and the buffer and adapted agent persist across them.
- **Control loop:** the current model-based controller chooses an action, the resulting transition is added to replay, SAE is updated without reward, and the next action is planned using the adapted encoding and frozen source models ([evaluation loop](https://github.com/yangsizhe/MoVie/blob/f721518846ffd1de4fdf09a4fc380b720ad04183/src/algorithms/tdmpc/src/eval_adaptation.py#L86-L116)).

The principal baselines isolate these choices: **TD-MPC** performs no adaptation; **DM** updates the encoder with dynamics loss but has no STNs; **IDM+STN** replaces forward dynamics supervision with inverse-action prediction and jointly adapts that inverse model, encoder, and STNs.

## Key results

Experiments cover 11 DMControl, four xArm, and three Adroit tasks. The paper reports three seeds (`0,1,2`) and 20 episodes per seed. DMControl values are cumulative reward; xArm and Adroit values are success percentages.

| Camera shift | DMControl: no adapt → MoVie | xArm: no adapt → MoVie | Adroit: no adapt → MoVie |
|---|---:|---:|---:|
| Novel view | 395.61 → **623.19** | 16 → **46** | 8 → **34** |
| Moving view | 605.98 → **673.74** | 20 → **42** | 15 → **45** |
| Shaking view | 441.79 → **558.23** | 42 → **45** | 30 → **63** |
| Novel FOV | 527.47 → **770.56** | 34 → **75** | 31 → **68** |
| All settings | 492.71 → **656.43** (+33%) | 28 → **52** (+86%) | 21 → **53** (+152%) |

The aggregate result hides meaningful task and shift dependence:

- **Reacher downstream control:** on DMControl Reacher-hard, no-adaptation → MoVie returns are `592.91±59.96 → 821.03±67.89` (novel), `854.76±31.64 → 872.46±51.30` (moving), `205.53±25.16 → 453.48±381.60` (shaking), and `424.11±83.74 → 707.75±86.05` (FOV). The source-view return is `937.43±54.59`; adaptation substantially recovers control under three shifts but does not fully restore it.
- **Manipulation evidence:** Adroit-door rises from 0% to 66% under both fixed novel and moving views, 1% to 83% under shaking, and 1% to 81% under novel FOV. Conversely, xArm-push changes only 46%→48% under novel view and 64%→57% under shaking. MoVie is not uniformly best on every task.
- **Shallow placement ablation:** on Cheetah-run, inserting two STNs is best among 0–4 STNs in all four shifts: novel `254.42→342.39`, moving `344.70→365.22`, shaking `317.66→493.54`, and FOV `379.01→532.94` when comparing zero versus two. Three or four STNs add no consistent benefit.
- **STN beyond encoder-only adaptation:** across all settings, DM versus MoVie is `520.68→656.43` on DMControl, `40→52` on xArm, and `30→53` on Adroit. This supports adapting shallow spatial features in addition to merely fine-tuning the encoder with the same forward-dynamics objective.
- **Frozen dynamics ablation:** freezing rather than fine-tuning dynamics gives all-setting scores of `64 vs 53` on xArm-push, `32 vs 6` on xArm-hammer, `959.51 vs 552.91` on Cup-catch, `747.31 vs 211.46` on Finger-spin, and `433.52 vs 410.68` on Cheetah-run. The advantage is large on four of five matched tasks, although fine-tuning wins isolated cells such as xArm-push novel view (`51 vs 48`) and Cheetah-run FOV (`561.94 vs 532.94`).

These are direct downstream-control evaluations, not only representation or prediction metrics. However, adaptation occurs during the same 20-episode evaluation stream, so the reported average mixes early adaptation and later adapted behavior.

## What is relevant for WhiteHole

WhiteHole explicitly studies frozen latent world models under visual shifts ([repository scope](../../../README.md)), making MoVie unusually close to its central hypothesis.

- **Strong evidence for freezing the predictive/control core:** MoVie changes perception while preserving the dynamics and control coordinate system. Its freeze-versus-fine-tune ablation supports WhiteHole's default of frozen backbones/predictors during adapter training ([current Two-Room freeze controls](../../../whitehole/adaptation/adapters.py)), though it does not prove this is optimal for JEPA models or appearance shifts.
- **Adaptation placement is a first-class variable:** MoVie adapts RGB and first-convolution features, not a late latent alone. WhiteHole already has both a small input-space Reacher adapter ([`SmallConvAdapter`](../../../scripts/reacher_conv_adapter.py)) and post-encoder Two-Room latent adapters. A matched placement comparison can therefore test whether geometric camera shifts require earlier adaptation than color/style shifts.
- **The loss is the WhiteHole one-step core:** MoVie's objective is the same structural constraint as WhiteHole's `dynamics_alignment_loss`, `||P(A(z_t),a_t)-A(z_{t+1})||²`, while WhiteHole additionally implements multi-step rollout and source-geometry terms ([objectives](../../../whitehole/adaptation/adapters.py)). MoVie supplies downstream evidence that a frozen action-conditioned predictor can supervise perception without reward, but not evidence that one-step consistency uniquely recovers source latents.
- **Dynamic views matter:** moving and shaking cameras change the observation map within a trajectory. This is a stricter test than one stationary target domain and directly probes whether frame-wise or state-dependent adapters can track nonstationary shifts.
- **Control must remain the final criterion:** MoVie demonstrates cases where adaptation changes task return/success dramatically and cases where it does not help. WhiteHole should not infer control recovery from lower latent loss alone; probe, rollout, and closed-loop planning/control metrics must remain separate.
- **Reacher transfer is suggestive, not direct:** the DMControl Reacher-hard gains are especially relevant to WhiteHole's Reacher track, but MoVie uses TD-MPC with 84×84 frame stacks and online MPC, whereas WhiteHole wraps an external LeWM stack. The result motivates a matched test; it is not a reproduction or architecture-independent guarantee.

## Concrete experiments to run next

1. **Frozen-core matrix (matched to the freeze-DM ablation).** On the same Two-Room and Reacher shifts, compare `(adapter only; adapter + source encoder; adapter + predictor/dynamics)` while keeping planner/policy, data, update count, and initialization fixed. Report one- and multi-step error, source-coordinate latent error where available, and closed-loop control. **WhiteHole hypothesis:** updating the predictive core will lower adaptation loss but damage source-compatible rollout/control more often than adapter-only training.

2. **Perception-placement ablation (matched to 0–4 STNs).** Under controlled camera displacement and FOV changes, compare parameter-matched adapters at RGB input, after the first convolution, both shallow locations, and post-encoder latent space. Include the existing residual input CNN and latent MLP/affine adapters. **WhiteHole hypothesis:** shallow adapters will dominate latent-only adapters for geometric camera shifts, while that advantage may disappear for pure appearance shifts.

3. **One-step versus rollout supervision (matched to MoVie's dynamics objective).** Freeze the same source predictor and train identical adapters with one-step loss, horizons `{4,8,15}`, and mixed one-/multi-step loss; use paired source alignment only as an oracle diagnostic. **WhiteHole hypothesis:** one-step loss is enough for static mild shifts but permits source-incompatible solutions that become visible in long rollouts and planning.

4. **Four camera-shift suite (matched to the paper's benchmark).** Render fixed novel pose, smooth moving trajectory, per-step shake at controlled amplitudes, and both larger/smaller FOV while holding physical state/action trajectories fixed. Evaluate at multiple severities and include a stationary appearance-shift control. **WhiteHole hypothesis:** moving and shaking shifts will expose failures hidden by fixed target-domain evaluation, especially for a single global latent transform.

5. **Reward-free online adaptation curve with downstream control (matched to the 20-episode protocol).** Let the deployed controller collect transitions, adapt without rewards, and record per-episode dynamics loss, rollout error, return/success, and interaction/update count. Compare no adaptation, encoder-only DM, and shallow-adapter DM. Report early and late episodes separately rather than only their average. **WhiteHole hypothesis:** prediction improvement will precede control recovery when alignment is valid, while divergence between them will identify dynamics-consistent but control-incompatible adaptation.

## Risks / open questions

- **No real-robot evidence:** all 18 tasks are simulated despite the robotics motivation.
- **2D affine limitation:** an STN can correct translation, rotation, scale, shear, and cropping, but cannot generally invert 3D viewpoint changes, reveal occluded content, or repair depth ambiguity.
- **Objective non-identifiability:** both sides of the one-step target use the trainable encoder, with no source-latent anchor or anti-collapse term. A frozen dynamics model constrains solutions empirically but does not guarantee recovery of the source coordinate system or control-relevant information.
- **Evaluation/adaptation coupling:** online data is generated by the currently adapted controller; poor early control changes the adaptation distribution. Averaging all 20 episodes does not isolate pre-adaptation, adaptation speed, and final performance.
- **Variance and aggregation:** only three seeds are used, several task-level dispersions are large, and domain averages combine tasks with very different source performance. The manuscript also calls the full suite both `18×4` and “64 configurations,” although `18×4=72`.
- **Ambiguous xArm shake specification:** the appendix gives xArm Gaussian standard deviation `0.4 m` but clips samples to only `±0.07 m`; this may be intentional heavy clipping or a decimal typo and should not be copied without checking the environment implementation.
- **Paper/code schedule mismatch:** the appendix says 32 updates after every interaction, but released configs ramp TD-MPC from 1 to 32 updates over 200 steps and MoDem from 1 to 16 over 100 steps ([TD-MPC config](https://github.com/yangsizhe/MoVie/blob/f721518846ffd1de4fdf09a4fc380b720ad04183/src/algorithms/tdmpc/cfgs/default_adaptation.yaml#L18-L21)). Reproduction should report the actual schedule.
- **Overlapping optimizers in released code:** the TD-MPC `encoder_optim` includes the inserted STNs, while dedicated STN optimizers also step them in the same update. This effectively applies overlapping Adam updates and should not be silently treated as the clean paper-level algorithm.
- **Compute and deployment cost:** the appendix reports roughly one hour per seed on an RTX 3090 despite under 2 GB memory. Thirty-two gradient updates per environment step may be unsuitable for real-time control without adaptation-rate and latency ablations.
