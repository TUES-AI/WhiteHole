# Learning and Leveraging World Models in Visual Representation Learning (Image World Models)

**Quentin Garrido, Mahmoud Assran, Nicolas Ballas, Adrien Bardes, Laurent Najman, Yann LeCun — 2024**  
**Primary source:** [arXiv:2403.00504v1](https://arxiv.org/abs/2403.00504v1) ([HTML](https://arxiv.org/html/2403.00504v1), [PDF](https://arxiv.org/pdf/2403.00504v1))

> **Source note.** This summary was prepared from the complete v1 source at `/tmp/arxiv-src/2403.00504/paper.tex`, including every table and the appendices on training/evaluation, augmentations, predictor tuning, scaling, out-of-domain evaluation, invariance, and qualitative retrievals. The source has no `\input`/`\include` children; the bibliography is in `paper.bbl`. Relevant source figures were also inspected. The manuscript contains no direct project-code link, so no implementation was imported.

## Problem and core idea

Most visual self-supervised methods retain the encoder but discard the predictor/decoder after pretraining. Image World Models (IWM) asks whether a JEPA predictor can instead learn a reusable **latent transformation model** and later be fine-tuned for downstream tasks.

IWM broadens I-JEPA's masked-patch prediction into prediction of both missing regions and global photometric changes. It treats a known image transformation as an “action”: given a corrupted/masked source image, target patch positions, and the parameters of the source-to-target transformation, the predictor estimates the EMA encoder's target-image latents. A predictor that can apply the transformation is called **equivariant**; one that cannot is called **invariant**. The paper's central empirical claim is that useful transformation prediction requires:

1. explicit conditioning on the transformation;
2. a prediction problem difficult enough that the predictor cannot ignore it; and
3. enough predictor capacity.

The resulting trade-off is important: weaker/invariant predictors encourage abstract, linearly accessible encoder features, while stronger/equivariant predictors preserve transformation-related information and become better reusable modules for predictor fine-tuning.

## Method details (short)

### Transformation prediction as a JEPA objective

Starting from one ImageNet image, IWM constructs a shared cropped/flipped image `I'`, then forms:

- **Target `y`:** color jitter only; destructive transformations are deliberately excluded so the target retains information.
- **Source `x`:** independently color-jittered, optionally blurred, solarized, or converted to grayscale, then masked with the union of four rectangles.
- **Action `a_{x→y}`:** color-jitter differences plus indicators for destructive source transformations. Mask-token positions provide the geometric part of the action.

A student ViT encoder produces `z_x = f_θ(x)` and an EMA teacher produces `z_y = f_θ^EMA(y)`. The predictor/world model receives visible source latents, target-position mask tokens, and `a_{x→y}`, and minimizes squared latent error on the paper's selected target-position index set:

`L(x,y) = Σ_{i∈M_x^C} ||p_φ(f_θ(x), a_{x→y}, m_a)_i - f_θ^EMA(y)_i||²`.

The default encoder is ViT-B/16, pretrained for 300 ImageNet epochs. Predictors are ViTs denoted `IWM^Inv/Equi_{depth,width}`. Default feature conditioning concatenates action scalars to each position-augmented mask token and mixes them through a three-layer MLP. Sequence conditioning instead inserts action tokens, with separate linear projections needed to break the transformer's permutation ambiguity.

The default source and target color jitter is applied with probability 0.8 (brightness 0.4, contrast 0.4, saturation 0.2, hue 0.1). Source-only blur, grayscale, and solarization probabilities are 0.2, 0.2, and 0.1 in the appendix's default table, although the adjacent prose says 0.2 for solarization too. The four masks each cover 15–20% of the image before overlap.

### Measuring whether transformations were learned

The paper uses mean reciprocal rank (MRR). For each image/transformation, it asks the predictor to transform a clean-image latent, ranks the desired transformed latent among a bank of 256 augmented-image latents, and averages reciprocal rank. MRR near 1 indicates that the predictor distinguishes and applies the requested transformation; MRR near 0 indicates invariance or failure.

This is transformation retrieval, not pixel reconstruction. Qualitative banks show smooth responses to brightness, contrast, saturation, and hue, but also failures when “inverting” grayscale—an intrinsically ambiguous operation. The authors explicitly use “equivariant” loosely because not every augmentation is a group action.

### Predictor fine-tuning (“adaptation” in IWM)

For downstream classification, the encoder is frozen and the pretrained predictor is placed on the **EMA teacher** features, asked to predict a full untransformed image using null transformation parameters, and trained with an attentive classification head. The default protocol fine-tunes for 100 ImageNet epochs with AdamW; the pretrained predictor learning rate is one tenth of that used for a randomly initialized predictor.

This is supervised repurposing of the predictor, not visual-domain adaptation. The predictor's pretraining objective is no longer itself the downstream goal; it supplies a useful initialization and latent-processing stack. The paper also studies encoder-only fine-tuning, end-to-end encoder-plus-predictor fine-tuning, semantic segmentation, and task-token-conditioned multitask predictor tuning.

## Key results

### What makes transformation prediction work

| Ablation | MRR | Interpretation |
|---|---:|---|
| No transformation conditioning | 0.00 | Predictor cannot apply the requested photometric action |
| Sequence conditioning | 0.82 | Explicit action tokens work |
| Feature conditioning (default) | 0.79 | Similar MRR and better downstream performance |

Predictor depth and task difficulty interact strongly:

| Transform setting | I-JEPA 12×384 | IWM 12×384 | IWM 18×384 |
|---|---:|---:|---:|
| Jitter | 0.00 | 0.11 | 0.25 |
| + destructive transforms | 0.00 | 0.09 | **0.79** |
| + strong jitter | 0.00 | **0.81** | **0.85** |

Across five color-containing appendix settings, the 18-layer predictor became color-equivariant in 4/5 settings, versus 1/5 for 12 layers. This result is not “stronger corruption is always better”: for the 18-layer model, a very strong jitter-plus-destructive setting reached MRR 0.85 but collapsed linear accuracy to 34.3% and predictor-fine-tuning accuracy to 81.7%, versus 67.5% and 83.3% under the default MRR-0.79 setting. Predictor capability, encoder abstraction, and downstream utility are distinct axes.

### Reusing the pretrained predictor

ImageNet-1k top-1 after 300-epoch ViT-B/16 pretraining:

| Model | Encoder fine-tuned, no predictor | Frozen encoder + random predictor | Frozen encoder + pretrained predictor | Encoder + predictor end-to-end |
|---|---:|---:|---:|---:|
| I-JEPA | 83.0 | 79.1 | 80.0 (+0.9) | 82.0 |
| `IWM^Inv_{12,384}` | **83.3** | 80.5 | 81.3 (+0.8) | 82.7 |
| `IWM^Equi_{18,384}` | 82.9 | 81.5 | **83.3 (+1.8)** | **84.4** |

The random-predictor control is the strongest evidence that transformation pretraining contributes more than merely adding a large head: the equivariant IWM gains 1.8 points from predictor pretraining, while MAE gains only 0.1–0.3 and the invariant IWM gains 0.8. Still, predictor tuning only slightly exceeds encoder tuning for the ViT-B equivariant model (83.3 vs. 82.9); the larger 84.4 result updates both components.

Other evidence is consistent with reuse of an equivariant predictor:

- **ADE20K segmentation:** `IWM^Equi_{18,384}` obtains 44.2 mIoU with encoder tuning, 46.8 with frozen-encoder predictor tuning, and 47.0 end-to-end. The invariant IWM obtains 45.6, 45.7, and 46.5.
- **ViT-L scaling:** `IWM^Equi_{36,512}` obtains 83.7 with encoder tuning, 85.0 with predictor tuning, and 85.4 end-to-end. The appendix recommends a predictor/encoder parameter ratio around 0.3, while warning that width and EMA settings become unstable when scaling.
- **Multitask predictor tuning:** one task-token-conditioned predictor averages 73.5 across ImageNet, iNat18, SUN397, and Places205 versus 73.4 for separately tuned predictors. This average hides task-specific changes from −1.2 on ImageNet to +2.6 on SUN397.
- **Representation abstraction:** invariant IWM is better at ImageNet linear/attentive probing (74.5/77.0) than equivariant IWM (67.5/75.1), whereas MRR correlates positively with predictor-tuning accuracy (`ρ=0.93`) and negatively with linear accuracy (`ρ=-0.85`) across the studied IWM settings.
- **Downstream transfer:** under attentive probing, equivariant IWM trails invariant IWM on ImageNet (75.1 vs. 77.0) but leads on iNat18 (54.2 vs. 51.6), SUN397 (71.7 vs. 71.0), and Places205 (60.5 vs. 59.4). These are dataset-transfer results, not controlled appearance-shift adaptation.
- **Marginalizing transformations is not enough:** averaging 8–128 predicted augmented latents does not improve linear top-1 over one prediction (64.3–64.6 vs. 64.5). Useful invariance appears to come from information removal during representation learning, not merely averaging an information-preserving equivariant model afterward.

## What is relevant for WhiteHole

WhiteHole studies frozen latent **temporal** world models under visual observation shifts, with small input/latent adapters and transition/rollout objectives ([repository scope](../../../README.md), [current latent-adapter objectives](../../../whitehole/adaptation/adapters.py), [Two-Room input adapter](../../../scripts/train_input_film_adapter.py)). IWM studies a static-image predictor jointly pretrained to apply known augmentation actions, then supervised-fine-tuned for recognition. The overlap is real, but limited.

### Where IWM supports WhiteHole's hypotheses

- **Latent predictors can model visual transformations rather than merely ignore them.** Conditioning changes MRR from 0.00 to 0.79–0.82. This directly supports the mechanism behind treating a known observation transformation or shift descriptor as an action/condition in latent space.
- **A pretrained predictor can contain reusable information beyond the encoder.** The pretrained-versus-random control (+1.8 ImageNet points for equivariant IWM), ADE20K results, and ViT-L results support the broad claim that discarding the JEPA predictor can waste transferable computation.
- **Freezing a visual encoder while adapting a downstream module can work.** IWM's strongest predictor-tuning protocol leaves the encoder frozen. This is directionally aligned with WhiteHole's modular adaptation premise and shows that frozen representations do not necessarily force linear probing.
- **Capacity must match transformation difficulty.** The depth/augmentation interaction supports WhiteHole's use of controlled mild/medium/strong shifts and capacity ablations. It also warns that failure of a tiny adapter or predictor does not establish impossibility.
- **Transformation sensitivity can preserve task-relevant detail.** Equivariant IWM has worse linear accessibility but better predictor tuning, segmentation, and transferred attentive probes. This supports WhiteHole's decision to evaluate latent alignment, state probes, rollouts, and planning separately instead of equating invariance or linear quality with world-model usefulness.
- **A single conditioned module can serve multiple tasks.** The multitask task-token result supports testing one shift-conditioned adapter rather than assuming a separate model per visual shift, though the IWM evidence concerns supervised tasks rather than shifts.

### Where IWM does **not** support WhiteHole's hypotheses

- **It does not freeze the whole source world model.** IWM fine-tunes the predictor itself; WhiteHole's central setup freezes both encoder and temporal predictor and learns an external input/latent adapter. The paper never tests that configuration.
- **It does not learn adaptation from target-domain transitions.** There is no dynamics-consistency, multistep-rollout, reward-free target adaptation, or source-coordinate recovery objective. Thus it does not validate WhiteHole's self-supervised adaptation losses.
- **Its “actions” are known synthetic image augmentations, not environment actions.** The IWM predictor maps one static view to another. It does not model physical state transitions, temporal compounding error, control, or planning.
- **It does not evaluate controlled observation shift.** ImageNet pretraining/fine-tuning and dataset transfer do not test a fixed underlying state rendered differently while preserving dynamics. No experiment adapts from one visual domain to another while retaining a frozen source predictor.
- **It does not show that ordinary JEPA is naturally appearance-invariant.** Unconditioned prediction encourages invariance, but I-JEPA's measured behavior lies between invariant and equivariant models and is not explicitly controlled. A JEPA can retain augmentation information; robustness to WhiteHole shifts remains empirical.
- **It does not show that transformation prediction improves control.** There are no rewards, policies, MPC rollouts, or planning-success measurements. Classification/segmentation improvements cannot establish that an adapted latent remains usable by a frozen planner.
- **It weakens any “more invariance is always better” hypothesis.** Invariance helps simple linear readout, but richer equivariant representations produce higher peak performance with a capable adapted predictor. Conversely, maximal MRR is not sufficient either: overly strong corruption can sharply damage representation quality.

The most defensible conclusion for WhiteHole is therefore: **IWM validates transformation-conditioned latent prediction and reuse of a pretrained predictor, but provides only indirect motivation—not evidence—for adapting a fully frozen temporal JEPA across visual shifts.**

## Concrete experiments to run next

1. **Known-shift conditioning on Two Rooms.** Train one shared input or latent adapter over mild/medium/strong shifts in three forms: no condition, discrete shift token, and oracle continuous transformation parameters. Freeze the encoder and temporal predictor in all cases, hold out intermediate severities and transformation compositions, and report paired source-latent MSE, source-probe RMSE, rollout error by horizon, and MPC success. This is the closest WhiteHole test of IWM's 0.00-versus-0.8 conditioning result.

2. **Frozen-component adaptation matrix with a random-module control.** Under identical Two-Room and Reacher shifts, compare (a) frozen encoder/predictor + input adapter, (b) frozen encoder/predictor + latent adapter, (c) frozen encoder + predictor fine-tuning as in IWM, and (d) adapter + predictor tuning. For (c), compare pretrained and randomly reinitialized predictors and parameter-match the trainable modules. This separates IWM-style predictor reuse from WhiteHole's stricter frozen-world-model claim.

3. **Transformation difficulty × capacity factorial.** Cross shift severity and composition with adapter capacities (channel affine, small residual CNN, latent affine/low-rank/MLP), using at least five seeds because IWM reports equivariance emerging in only 1/5 or 4/5 runs depending on predictor depth. Track not just final loss but active latent dimensions, source/target covariance, transformation-retrieval MRR, horizon-wise rollout error, and planning. Test for an intermediate optimum rather than assuming that stronger shifts or larger modules always help.

4. **IWM-augmented temporal JEPA pretraining.** Train matched source models with (a) temporal action-conditioned prediction only, (b) temporal prediction plus unconditioned photometric augmentation, and (c) temporal prediction plus explicit photometric-action conditioning. Freeze each source model, fit the same small adapter on held-out visual shifts, and compare source-domain dynamics, target adaptation, and MPC. This tests whether learning visual equivariance during source pretraining makes later frozen-model adaptation easier or instead preserves nuisance detail that the adapter must remove.

5. **Does IWM's MRR predict WhiteHole success?** For paired synthetic shifts, build 256-candidate source-render latent banks and measure whether the shifted or adapted latent retrieves the matching state; separately measure whether a shift-conditioned predictor retrieves the requested transformed state. Correlate MRR with paired latent error, source-probe transfer, multistep rollout error, and planning success across shifts/capacities/seeds. This tests whether transformation prediction is a useful model-selection diagnostic or merely a static retrieval score.

6. **Task-token-style multi-shift adapter.** Compare separate adapters per shift with one shared adapter conditioned by learned shift tokens, balancing samples and trainable parameter counts. Evaluate seen shifts, held-out severities, and mixed batches. Report both average and per-shift results so an IWM-like unchanged average cannot hide large regressions on particular shifts.

## Risks / open questions

- **Different meanings of “world model”:** IWM's predictor models synthetic transformations of a static image; WhiteHole's predictor models action-conditioned temporal dynamics. A successful mechanism may not transfer between them.
- **Known actions versus unknown domain shifts:** IWM supplies exact augmentation metadata. Real target observations may not expose a shift label, continuous severity, or invertible transformation parameters.
- **Irreversible transformations:** grayscale, blur, masking, and occlusion destroy information. Apparent inversion can exploit dataset regularities or latent invariance rather than recover the true missing state; the qualitative grayscale failures show this limit.
- **MRR is a constrained metric:** it ranks within only 256 generated candidates and does not establish calibrated prediction, compositional generalization, temporal consistency, or source-coordinate identification.
- **Optimization and statistical uncertainty:** several headline tables report selected hyperparameters without confidence intervals, and the appendix itself reports large seed sensitivity in whether a 12- or 18-layer predictor becomes equivariant.
- **Fine-tuning is not necessarily lightweight at inference:** the 18- or 36-layer predictor remains in the deployed path. It may reduce trainable parameters and amortize a frozen backbone across tasks, but it adds substantial inference capacity compared with WhiteHole's tiny adapters.
- **Supervision mismatch:** IWM predictor adaptation uses labeled classification/segmentation objectives for many epochs. It gives no evidence that unlabeled transition consistency can choose the correct visual-to-source mapping.
- **Non-identifiability remains:** matching a transformation or frozen dynamics predictor need not recover source latent coordinates needed by a source probe or planner. WhiteHole must retain paired-oracle diagnostics in controlled environments and planning as a separate consequential test.
- **No official implementation was linked in the manuscript:** exact reproduction would require reconstructing details not fully captured by the paper, including transformation-parameter encoding and some predictor-fine-tuning plumbing. The source also has reproducibility ambiguities: its prose and default augmentation table disagree on solarization probability (0.2 vs. 0.1), and its mask/complement notation is difficult to reconcile with the verbal description of dropped and predicted patches.
