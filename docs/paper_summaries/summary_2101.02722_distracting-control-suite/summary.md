# The Distracting Control Suite — A Challenging Benchmark for Reinforcement Learning from Pixels

**Paper:** Austin Stone, Oscar Ramirez, Kurt Konolige, and Rico Jonschkowski, arXiv:2101.02722v1 (2021) — [arXiv](https://arxiv.org/abs/2101.02722v1) · [TeX source](https://arxiv.org/src/2101.02722v1) · [official implementation at the paper-era commit](https://github.com/google-research/google-research/tree/0c1bbe5fc971a1de1a427debc814e66ab4f1e7fa/distracting_control)

This summary is based on the complete `paper.tex`, its figures, tables, and bibliography, and is cross-checked against official code commit [`0c1bbe5`](https://github.com/google-research/google-research/commit/0c1bbe5fc971a1de1a427debc814e66ab4f1e7fa), dated two days before the arXiv submission. All quantitative results below are **paper claims, not WhiteHole results**.

## Problem and core idea

Standard pixel-based DM Control has a fixed camera, fixed object appearance, and static background, so nearly every visual change is tied to task state. The paper introduces the **Distracting Control Suite (DCS)** to vary task-irrelevant rendering while leaving physics, rewards, and control properties unchanged. Its contribution is a configurable benchmark and an empirical diagnosis of then-current pixel-RL baselines—not an adaptation algorithm.

DCS has three independently composable distraction axes:

1. **Camera pose:** random camera azimuth/elevation, distance, and roll. The camera continues to look at the agent for tracking-camera tasks or the original scene focus for fixed-camera tasks; pitch/yaw are derived for that look-at behavior rather than independently randomized.
2. **Object color:** independently perturbed RGB material channels, bounded around each original color.
3. **Background:** natural DAVIS 2017 video frames projected onto the MuJoCo skybox, with task-dependent floor transparency so the background is visible.

Every axis can be **static**—sampled at episode reset and then held—or **dynamic**—initialized at reset and changed smoothly within the episode. Axes can be evaluated alone or in combination, and their severities are controllable. This creates observation shifts without changing the underlying transition system, which is exactly the invariance WhiteHole wants a frozen world model to exploit.

## Method details (short)

### Exact distraction definitions

| Axis | Episode initialization / static setting | Dynamic setting | Severity |
|---|---|---|---|
| **Camera** | Uniformly sample a pose within the allowed angular, roll, and radial range; hold it for the episode. | Start from the same pose distribution, then evolve translation and roll velocities by clipped Gaussian random walks. | For `β_cam∈[0,1]`, angular and roll deltas scale as `πβ_cam/2`; radius spans `r₀(1−0.5β_cam)` to `r₀(1+1.5β_cam)`. At `β=1`, this reaches `0.5r₀–2.5r₀` and up to `90°` angular/roll displacement. Dynamic limits are `v_max=0.4β`, `σ_v=0.1β`, `v_roll,max=πβ/50`, and `σ_roll=πβ/300`. |
| **Color** | For every RGB channel with original value `x`, sample uniformly in `[x−β_rgb,x+β_rgb]`, clipped to valid RGB bounds; hold it. | Add per-step Gaussian noise with standard deviation `0.03β_rgb`, clipping each channel to remain within `β_rgb` of its original value. | `β_rgb∈[0,1]` is the maximum per-channel deviation. |
| **Background** | Choose one of the first `b` DAVIS training videos and one random frame at reset; hold that frame. | Choose a random video, start frame, and direction; play frame by frame, reversing at either endpoint to avoid cuts. | Usually the number of available videos `b∈[0,60]`. One clean-trained sensitivity experiment instead fixes `b=60` and blends DAVIS with the original skybox using opacity `β_bg∈[0,1]`. |

The official implementation confirms the benchmark presets and dynamics: [`easy=0.1`, `medium=0.2`, `hard=0.3`; 4, 8, or all background videos](https://github.com/google-research/google-research/blob/0c1bbe5fc971a1de1a427debc814e66ab4f1e7fa/distracting_control/suite_utils.py#L26-L54), explicit [DAVIS train/validation video lists](https://github.com/google-research/google-research/blob/0c1bbe5fc971a1de1a427debc814e66ab4f1e7fa/distracting_control/background.py#L28-L47), and [static-frame versus ping-pong video playback](https://github.com/google-research/google-research/blob/0c1bbe5fc971a1de1a427debc814e66ab4f1e7fa/distracting_control/background.py#L165-L175). The paper itself reports only **easy** and **medium** combined benchmarks; the code's `hard` preset is not a reported paper benchmark.

A specification subtlety matters for reproduction: the manuscript describes camera variables over an upper-frontal range, while the code samples horizontal/vertical angles and roll as symmetric deltas around the original pose, then constrains non-Reacher tasks to the upper quadrant ([pose sampling](https://github.com/google-research/google-research/blob/0c1bbe5fc971a1de1a427debc814e66ab4f1e7fa/distracting_control/camera.py#L212-L270)). Use the versioned implementation rather than reconstructing signs from the paper's notation.

Background projection also changes floor visibility: opacity is `0` for Reacher, `1` for Walker/Cheetah, and `0.3` otherwise ([code](https://github.com/google-research/google-research/blob/0c1bbe5fc971a1de1a427debc814e66ab4f1e7fa/distracting_control/suite_utils.py#L57-L80)). Thus “background only” is not literally only a skybox texture change relative to vanilla DM Control.

### Tasks, baselines, and protocol

The benchmark uses six DM Control tasks with Planet-style action repeats:

| Task | Action repeat |
|---|---:|
| Ball-in-Cup Catch | 4 |
| Cartpole Swingup | 8 |
| Cheetah Run | 4 |
| Finger Spin | 2 |
| Reacher Easy | 4 |
| Walker Walk | 2 |

Baselines are model-free **SAC** and **QT-Opt**, each without augmentation, with one random crop per sample (**RAD**), or with two-crop averaging in target and Q estimates (**DrQ**, expressed as `K=M=2`). All share the DrQ four-layer CNN and a 50-dimensional normalized visual representation. Training uses batch size 512, one gradient step per collection step, 500K environment steps, and five random seeds per task unless stated otherwise. Final evaluation uses 100 episodes; tables report means and standard errors.

There are three distinct protocols that should not be conflated:

1. **Clean-train / distracted-test sensitivity:** train on ordinary DM Control, then evaluate one distraction axis at a time over severity `0…1`, in both static and dynamic modes. Background uses all 60 training videos while varying blend opacity `β_bg`.
2. **Distracted-train / distracted-test sweeps:** train and evaluate with one distraction axis. Camera and color vary their `β`; backgrounds are fully opaque while `b` varies. A separate background-generalization test trains on DAVIS training videos and evaluates on all 30 unseen DAVIS validation videos.
3. **Combined benchmark:** train and evaluate with camera + color + background together. **Easy** is `β_cam=β_rgb=0.1, b=4`; **medium** is `0.2, 0.2, b=8`. Both have static and dynamic variants. A **blind** lower bound uses medium parameters but points the camera away from task-relevant objects.

This is conventional end-to-end RL training under distractions. It does **not** freeze an encoder or dynamics model, adapt at deployment, use reward-free target transitions, or distinguish an adaptation buffer from final evaluation data.

## Key results

- **Unseen distractions break clean-trained agents at low severity.** Averaged over tasks, the strongest methods lose half their clean score around `β_cam=0.2`, `β_rgb=0.6`, and `β_bg<0.1`. Background replacement is therefore much more damaging than color variation under this blend-based comparison. Static versus dynamic test distractions make little difference when neither appeared during training.
- **Training on distractions helps unevenly.** It improves camera robustness and especially performance on seen backgrounds, but provides little improvement for color. For backgrounds, increasing the number of training scenes lowers performance on those scenes; unseen-video performance initially improves but then plateaus. Dynamic camera/background training often outperforms static training, plausibly because it exposes the learner to more visual variation per trajectory—not necessarily because dynamic shifts are intrinsically easier.
- **The combined benchmark is severe even at “easy.”** Mean return across six tasks falls from roughly `801–836` for the crop-augmented methods without distractions to the following:

| Combined setting | SAC+RAD | SAC+DrQ | QT-Opt+RAD | QT-Opt+DrQ |
|---|---:|---:|---:|---:|
| Easy, static | 182 | 166 | **317** | 299 |
| Easy, dynamic | 270 | 199 | **343** | 265 |
| Medium, static | 113 | 126 | 165 | **170** |
| Medium, dynamic | 89 | 89 | **103** | 102 |

  In medium-dynamic, methods only barely outperform the blind baseline. Sensitivity is strongly task dependent: locomotion tasks are generally harder than Ball-in-Cup, Cartpole, Finger, or Reacher.
- **Distractions interact super-multiplicatively.** If each single-axis score is normalized by clean performance, multiplying the three ratios consistently overestimates actual combined performance. For QT-Opt+RAD, the individual-effect product versus observed benchmark ratio is `0.48 vs 0.39` on easy-static and `0.39 vs 0.13` on medium-dynamic. The failure is therefore not explained by independent per-axis degradation.
- **Method rankings change.** Random cropping remains essential, but does not solve DCS. SAC- and QT-Opt-based augmented agents are comparable on clean DM Control; QT-Opt variants are strongest under combined distractions. RAD is generally equal to or better than the more expensive two-crop DrQ variant in these experiments.

These results establish benchmark difficulty, not why representations fail. Returns do not identify whether failures arise from latent geometry, state aliasing, value/policy extrapolation, or optimization.

## What is relevant for WhiteHole

WhiteHole currently has one controlled **fixed hard-camera** Reacher test: the source physics is retained, one target camera is used, and offline reward-free adaptation is evaluated with frozen LeWM dynamics/control. DCS adds several missing dimensions:

- **Three orthogonal shift axes:** geometry (camera), bounded appearance (color), and high-dimensional natural clutter/motion (DAVIS), rather than one camera transform.
- **Episode-level versus within-episode shift:** static randomization tests adaptation to a sampled target domain; dynamic camera/color/video tests whether the observation map changes along a rollout. A single fixed affine adapter can succeed on WhiteHole's current camera yet fail this nonstationary case.
- **Calibrated sweeps and compositions:** easy/medium levels and isolated/pairwise/three-way combinations can reveal thresholds and interactions hidden by a single “hard” point. DCS specifically warns that passing each shift independently does not predict composed-shift control.
- **A real held-out visual split:** DAVIS training versus validation videos supports a cleaner distinction between fitting encountered nuisance instances and generalizing to new ones. Camera and color sweeps, by contrast, sample from parameter ranges rather than semantically disjoint test sets.
- **More task diversity:** six tasks vary morphology, motion, camera mode, and action repeat, reducing the chance that a Reacher-only result is mistaken for general adaptation.

The transfer is nevertheless limited. DCS's original agents jointly learn representation and policy from rewards for 500K steps under the distraction distribution. WhiteHole adapts a small perception path against a **frozen JEPA/action-conditioned predictor**, with no target reward, then judges planning/control compatibility. DCS supplies shifts and evaluation structure, but no adapter, frozen-dynamics objective, sample-efficiency result, or evidence that any RL baseline will rank similarly in WhiteHole. Its returns cannot be treated as adaptation baselines without rebuilding the protocol around frozen-core, reward-free adaptation.

## Concrete experiments to run next

1. **DCS camera severity × temporal mode on the existing Reacher stack.** For `β_cam∈{0.1,0.2,0.3}`, compare one fixed target camera, DCS static per-episode poses, and DCS dynamic random-walk poses using unadapted, STN-only, full-update, and SWD-preserved checkpoints. Match trajectories and report source retention, one-/multi-step prediction, and closed-loop success. This tests whether the current `82%` fixed-camera STN-only result survives pose diversity and within-rollout motion.

2. **Single, pairwise, and triple distraction factorial.** At easy and medium severity, evaluate camera, color, background, all three pairs, and all three together with identical adaptation/control budgets. Compare actual composed success with the product of normalized single-axis success, mirroring the paper's interaction analysis. This can determine whether WhiteHole's frozen-core adapter also exhibits super-multiplicative failures.

3. **Held-out background adaptation.** Adapt without rewards on transitions using the first `{4,8}` DAVIS training videos; test on seen training videos and all 30 validation videos, separately for static frames and dynamic playback. Include no-adaptation and target-trained oracle controls, plus clean-source retention. Measure whether frozen dynamics learns task-relevant invariance or merely memorizes nuisance scenes.

4. **Severity extrapolation matrix.** Adapt at one severity (`0.1` or `0.2`) and test all severities through `0.3`, including unseen compositions. Keep update count and samples fixed. This is stricter than adapting and testing from the same range and exposes whether the adapter restores source-compatible coordinates or only fits one nuisance distribution.

5. **Offline versus interaction-coupled dynamic adaptation.** Under moving camera and video backgrounds, compare WhiteHole's fixed pre-collected target buffer with MoVie-style online reward-free updates during control. Plot performance and dynamics loss by episode rather than averaging the whole stream. Dynamic DCS provides the needed test of whether stale offline transitions are sufficient when the observation map changes continuously.

## Risks / open questions

- **Benchmark, not method:** DCS offers no prescription for frozen-JEPA adaptation and no evidence about adapter placement, predictor freezing, collapse, or source-coordinate identifiability.
- **Old, mismatched baselines:** SAC/QT-Opt with shallow CNNs and random crops are end-to-end model-free RL baselines. Their ranking and 500K-step returns are not directly comparable to a pretrained frozen world model with MPC.
- **Severity is not cross-axis calibrated:** `β_cam`, `β_rgb`, background count `b`, and blend opacity `β_bg` have different semantics. “Medium” is a preset, not evidence of equal perceptual or control difficulty across axes or tasks.
- **Static/dynamic is confounded with diversity:** dynamic agents see more nuisance states per episode. Better dynamic-training results do not isolate temporal robustness from increased augmentation exposure.
- **Observation sufficiency can fail:** large camera changes can create occlusion or remove state information; moving natural backgrounds can dominate pixels. Frozen dynamics cannot reconstruct information absent from the observation, so failure need not imply poor adaptation.
- **Rendering is not perfectly invariant:** task physics stays fixed, but floor transparency changes by task, and severe camera shifts alter visibility. These details should be versioned and reported rather than described generically as “same environment.”
- **Only backgrounds have an explicit held-out semantic split:** camera/color train and test draws can come from the same bounded family. Generalization claims require held-out ranges, trajectories, or transformation compositions beyond the paper's default protocol.
- **Paper/code preset mismatch:** official code exposes a `hard` preset (`β=0.3`, all training videos), while the paper defines and reports combined results only for easy and medium. WhiteHole should label any hard-preset result as a new experiment, not a paper reproduction.
