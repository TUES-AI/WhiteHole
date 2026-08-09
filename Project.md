# WhiteHole
## Status

WhiteHole is early exploratory research on adapting JEPA representations and latent world models under observation shift.
No final architecture, objective, adapter placement, benchmark, or paper claim has been selected.
Current work is intended to establish what works, what fails, and which diagnostics predict downstream behavior.

## Research question

Can a small trainable module make shifted observations usable by a pretrained or frozen JEPA without destroying useful latent geometry or control behavior?
The main variables are the backbone, adapter architecture, adapter placement, supervision, loss, data budget, and shift type.
A successful method should retain source-domain behavior, generalize beyond its training transformations, and use materially fewer trainable parameters or samples than full fine-tuning.

## Working hypotheses

- Early adapters may be better for geometric or camera shifts because they can repair spatial structure before encoding.
- Late adapters may be cheaper but can be non-identifiable relative to a frozen predictor or downstream head.
- Paired feature alignment is a strong controlled baseline, not proof of adaptation from naturally available supervision.
- Dynamics consistency may help without labels, but low prediction loss does not uniquely recover source-compatible coordinates.
- Multi-step objectives may expose failures hidden by one-step alignment.
- Adapter quality must ultimately be judged by the task that consumes the representation.

## Experimental tracks

### Frozen image representations

Use a frozen pretrained I-JEPA encoder with clean/transformed image pairs.
Compare input residual adapters, patch-embedding adapters, selected-block modules, post-encoder adapters, LoRA, and fine-tuning controls.
Compare global feature alignment, correspondence-aware patch alignment, multi-layer distillation, masked prediction, and supervised upper bounds.
Evaluate held-out transformations and severities, clean retention, retrieval, feature geometry, probes, sample efficiency, parameters, and runtime.

### Action-conditioned environments

Start with Reacher camera rotation and translation, then random per-episode poses and cameras that move within an episode.
Render source and target cameras from the same simulator state when paired supervision is needed.
Measure one- and multi-step prediction, state probes, source-coordinate alignment, and closed-loop control or MPC.
Treat occlusion and partial observability as possible limits of framewise adaptation, not implementation details to hide.

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
- [MoVie](docs/paper_summaries/summary_2307.00972_movie/summary.md) is a close precedent for reward-free perception adaptation with frozen dynamics and control.
- [AdaJEPA](docs/paper_summaries/summary_2606.32026_adajepa/summary.md) provides an online fine-tuning comparison across encoder, predictor, and LoRA placements.

## Near-term work
Establish frozen I-JEPA paired-transformation baselines before expanding the architecture and objective matrix.
Then run matched Reacher placement and loss comparisons under fixed and moving cameras.
Use PLDM environments only where they isolate a scientific question more cleanly than the main tracks.
Do not claim general adaptation, architectural novelty, or a final method until repeated downstream evidence supports it.
