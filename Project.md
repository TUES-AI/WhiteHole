# WhiteHole
## Status

WhiteHole is early exploratory research on adapting JEPA representations and latent world models under observation shift.
No final architecture, objective, adapter placement, benchmark, or paper claim has been selected.
MoVie is now the primary methodological starting point; the exact JEPA adaptation method and claim remain open.

## Research question

How should MoVie's frozen-dynamics test-time adaptation principle be transferred to JEPA world models without destroying useful latent geometry or control behavior?
The main variables are the backbone, adapter architecture, adapter placement, supervision, loss, data budget, and shift type.
A successful method should retain source-domain behavior, generalize beyond its training transformations, and use materially fewer trainable parameters or samples than full fine-tuning.

## Working hypotheses

- Early adapters may be better for geometric or camera shifts because they can repair spatial structure before encoding.
- Late adapters may be cheaper but can be non-identifiable relative to a frozen predictor or downstream head.
- Paired feature alignment is a strong controlled baseline, not proof of adaptation from naturally available supervision.
- Dynamics consistency may help without labels, but low prediction loss does not uniquely recover source-compatible coordinates.
- Multi-step objectives may expose failures hidden by one-step alignment.
- Time-varying cameras add observation dynamics that a framewise affine adapter may not be able to canonicalize, even when the physical dynamics and task remain unchanged.
- Adapter quality must ultimately be judged by the task that consumes the representation.

## Experimental tracks

### Frozen image representations

Use a frozen pretrained I-JEPA encoder with clean/transformed image pairs.
Compare input residual adapters, patch-embedding adapters, selected-block modules, post-encoder adapters, LoRA, and fine-tuning controls.
Compare global feature alignment, correspondence-aware patch alignment, multi-layer distillation, masked prediction, and supervised upper bounds.
Evaluate held-out transformations and severities, clean retention, retrieval, feature geometry, probes, sample efficiency, parameters, and runtime.

### Action-conditioned environments

Reproduce MoVie's frozen-dynamics adaptation logic on Reacher, then compare it with JEPA-specific adapters.
Use Distracting Control Suite camera, color, DAVIS-background, and composed shifts in both static and within-episode dynamic settings.
Compare spatial transformers, generic input adapters, latent adapters, and controlled encoder/predictor updates.
Measure severity extrapolation, held-out backgrounds, source-coordinate alignment, and closed-loop control or MPC.

### Reference environments

PLDM's Two-Room and related environments remain useful controlled testbeds and baselines.
PLDM is a reference system and environment source rather than WhiteHole's primary implementation foundation.
Easy invertible appearance shifts are calibration checks; they are not evidence of broad domain adaptation.

## Evaluation principles

- Separate source-oracle, shifted-unadapted, adapted, and target-trained upper-bound conditions.
- Match trainable parameters, data, optimizer steps, planner compute, and evaluation budgets where possible.
- Report multiple seeds and uncertainty rather than selected successful runs.
- Test held-out transformations, compositions, severities, and changing camera conditions.
- Measure clean-domain degradation alongside target-domain recovery.
- Keep representation, rollout, probe, and downstream behavior metrics separate.
- Require closed-loop control or planning evidence for claims about adapted world models.
- Record failures and non-identifiability, not only the best architecture.

## Paper foundations

- [A Path Towards Autonomous Machine Intelligence](docs/paper_summaries/summary_openreview_BZ5a1r-kVsf_autonomous-machine-intelligence/summary.md) frames JEPA, predictive world models, and planning as a research program rather than an established complete system.
- [LeJEPA](docs/paper_summaries/summary_2511.08544_lejepa/summary.md) motivates SIGReg as an anti-collapse alternative whose value for adaptation remains untested.
- [LeWorldModel](docs/paper_summaries/summary_2603.19312_leworldmodel/summary.md) provides a compact end-to-end JEPA world-model baseline and emphasizes planning evaluation.
- [PLDM](docs/paper_summaries/summary_2502.14819_pldm/summary.md) supplies reference environments, objectives, and reward-free planning comparisons.
- [Image World Models](docs/paper_summaries/summary_2403.00504_image-world-models/summary.md) studies transformation-conditioned latent prediction and predictor reuse.
- [MoVie](docs/paper_summaries/summary_2307.00972_movie/summary.md) is the primary adaptation precedent; [Distracting Control](docs/paper_summaries/summary_2101.02722_distracting-control-suite/summary.md) supplies the multi-axis benchmark.
- [SCMA](https://arxiv.org/abs/2502.09923) broadens this line with a policy-agnostic image denoiser and a frozen generative world model.
- [AdaJEPA](docs/paper_summaries/summary_2606.32026_adajepa/summary.md) is the closest JEPA-specific online adaptation comparison, but updates the world model itself.
## Near-term work
A matched offline MoVie-style Reacher ablation now separates visual adaptation scope from preservation objectives while freezing LeWM dynamics and control. On 100 identical starts, unadapted LeWM scores 85% on source and 35% on hard camera. Full STN+encoder+projector adaptation scores 78%/75%; STN-only scores 77%/82%; STN+encoder with a frozen projector scores 80%/73%; adding source identity to the full update scores 83%/70%; and adding source-relative latent, predicted-transition, and joint SWD preservation scores 80%/78%.
The current best hard-camera point estimate is therefore the 83,372-parameter STN-only adapter. Updating the encoder reduced hard-camera success by 9 points relative to STN-only on paired starts (95% bootstrap CI -17 to -1; exact McNemar `p=0.049`) despite improving held-out one-step dynamics MSE. Distribution preservation recovered most of that downstream loss and gave a balanced 80%/78% source/hard result, but did not statistically separate from STN-only at 100 cases. Direct source identity preserved source behavior best but traded away target success.
A harder dynamic Distracting Control test places the DAVIS `bear` sequence in Reacher's MuJoCo sky texture. With adaptation and evaluation separated by episode, unadapted source/bear success was 82%/18%. Full visual adaptation, STN-only, and STNs+encoder scored 13%/9%, 6%/16%, and 13%/8%: all reduced bear one-step dynamics MSE, none recovered bear control, and all destroyed source control. This establishes the tested affine MoVie adapter as a useful negative-control ablation for semantic dynamic backgrounds; it does not establish universal MoVie failure.
A 97,731-parameter coordinate-aware residual U-Net demonstrates that small non-affine input adaptation has enough capacity for partial bear recovery under strong paired supervision. On a screening-episode-excluded 100-case holdout, unadapted and STN-only source/bear success were 73%/19% and 16%/10%; two U-Net training seeds scored 80%/29% and 86%/35%. Both significantly beat STN-only on bear control, while the second also beat unadapted bear control. Target-only dynamics training of a parameter-matched residual CNN failed during screening; the successful U-Net uses exact paired source renders, source-latent alignment, and a heavily weighted source-pixel reconstruction loss. It is therefore an architecture/supervision upper bound, not a deployable test-time-adaptation method.
The experiments still use fixed target-transition buffers rather than interaction-coupled online MoVie, and the bear work uses a newly collected internally matched random-policy dataset because the collaborator's private cache was unavailable.
Next replace paired source pixels with an available source-domain prior or frozen generative/distribution-matching objective, with SCMA as the strongest methodological reference. Add static-bear and held-out-video tests after target-only recovery appears, then repeat competitive rows over adaptation seeds.
Retain the medium-shift full-encoder result as a historical comparison rather than a MoVie result, and keep frozen I-JEPA and PLDM as controlled diagnostic tracks.
These are promising controlled results, not yet a faithful online MoVie reproduction: training uses a fixed target-transition buffer before evaluation rather than interactions inside the control loop. All adapted rows also use one training seed and one regularization setting.
Next repeat the STN-only, full-update, and SWD rows over multiple training seeds; sweep SWD strength around the current `0.05/0.1/0.1` setting; isolate RGB-input versus patch-grid STN placement; and then separate adaptation interactions from final evaluation.
A new `dynamic_camera` Reacher benchmark now orbits the camera every 24 environment steps while changing elevation, radius, field of view, and look-at point from a deterministic per-episode phase. Live and goal images use the same instantaneous camera. The arm remains visible throughout the inspected trajectory, but unadapted, fixed-camera STN zero-shot, and same-budget dynamic-trained STN-only control score only 3/30, 4/30, and 4/30 respectively. Dynamic STN training reduces held-out dynamics MSE from `0.4676` to only `0.2883`, versus `0.2664` to `0.0876` on the fixed hard camera.
A camera-calibrated projective oracle now shows that canonicalization is the right first mechanism for this stress test. On 100 matched starts, raw dynamic-camera LeWM scores 12%, an exact dynamic-to-source homography scores 64%, and the source view scores 85%. The homography gain is +52 points (paired 95% bootstrap CI +41 to +63; exact `p=2.26e-13`). Filling unseen pixels with a state-independent source background also scores 64%, despite reducing paired pixel MSE, so background completion is not the remaining control bottleneck. The next experiment should learn the eight-degree-of-freedom projective transform, first from simulator pose supervision as a learned upper bound and then from short temporal windows using frozen dynamics and inverse-dynamics constraints. A factorized physical/camera latent model remains the next fallback if learned canonicalization cannot approach the calibrated oracle.
Retain the medium-shift full-encoder result as a historical comparison rather than a MoVie result.
Use PLDM environments only where they isolate a scientific question more cleanly than the main tracks.
Do not claim general adaptation, architectural novelty, or a final method until repeated downstream evidence supports it.
