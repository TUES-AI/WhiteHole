# Senior-review report: visual shifts with latent scene state fixed

**Date:** 2026-08-15  
**Scope:** Frozen I-JEPA, V-JEPA, and V-JEPA 2/2.1 encoders; controlled visual changes; layerwise analysis; possible benchmark contribution.

## Verdict

**Reject the broad formulation “benchmark JEPA robustness to visual shifts.”** It is already crowded. [SemanticMoments / SimMotion](https://arxiv.org/abs/2602.09146) already benchmarks V-JEPA 2, VideoMAE, TimeSformer, and other encoders on retrieval of the same motion across changed subject, foreground attributes, scene style, static context, and viewpoint. [Alrasheed et al. 2026](https://arxiv.org/abs/2605.15618) separately compare frozen V-JEPA 2/2.1 against VideoMAE-v2 and VideoPrism under pixel corruptions, occlusion, and temporal perturbations. Generic paired retrieval, corruption accuracy, cosine similarity, or JEPA-versus-VideoMAE comparisons would be incremental.

**Conditionally support a narrower paper on exact renderer-state counterfactuals and internal functional use.** The defensible gap is:

> **Under paired counterfactual observations that preserve latent scene state, geometry, camera, and dynamics, which visual factors remain decodable in frozen JEPA representations across depth, and when do those factors stop perturbing physics/content-relevant readouts and native predictor behavior?**

This framing is stronger than “appearance-discard zone” because it does not presuppose deletion, a unique zone, or JEPA superiority. The likely contribution is a **factor-controlled representation audit**, not another corruption leaderboard.

A public-checkpoint comparison can establish how specific model families behave. It cannot show that latent prediction *causes* the behavior unless the study also trains an architecture/data/compute-matched JEPA-versus-pixel control.

## Assessment of `prompt-idea.md`

The strongest part is the distinction between **information being decodable** and **information being functionally used**. Layerwise appearance and content probes alone are not enough; the clean contribution is to determine whether appearance can still be read out while paired appearance changes no longer alter task-relevant computation.

Three parts need reframing:

1. **“Appearance-Discard Zone” presupposes the result.** Use it only if appearance decodability actually collapses. The neutral question is appearance sensitivity, accessibility, and functional decoupling across depth.
2. **The current transformation list conflates different problems.** Texture, albedo, lighting, background, sensor corruption, occlusion, camera/viewpoint, crop, rotation, and shear do not all mean appearance. They should be separate benchmark families.
3. **Probe-subspace replacement should not be the first pillar.** It can be off-manifold and can remove correlated content. Exact paired re-rendering is already a clean causal intervention on the input. Add internal interventions only after a reproducible phenomenon exists and with completeness/selectivity controls.

A good title-level question is: **When nothing physical changes, what changes inside a visual world model?**

## The closest prior work and novelty threats

### 1. SimMotion already tests same motion under changed visual factors

[SemanticMoments: Training-Free Motion Similarity via Third Moment Features, arXiv:2602.09146](https://arxiv.org/abs/2602.09146) introduces [SimMotion-Synthetic](https://huggingface.co/datasets/Shuberman/SimMotion-Synthetic): 250 reference/positive/hard-negative triplets, with 50 examples for each of static-object, dynamic-appearance, dynamic-object, scene-style, and viewpoint changes. The positive is intended to preserve motion while changing the designated visual factor; the negative preserves appearance while changing motion. Its public evaluation includes V-JEPA 2, VideoMAE, TimeSformer, VideoPrism, SlowFast, I3D, and other RGB/flow/text baselines. On average retrieval accuracy, final V-JEPA 2 features score 74.4%, VideoMAE 79.2%, and TimeSformer 58.4%; their training-free temporal-moment readout raises the V-JEPA 2 score to 84.4%. [SimMotion-Real](https://huggingface.co/datasets/Shuberman/SimMotion-Real) adds 40 human-curated real-world triplets with 1,000 Kinetics distractors.

This directly occupies **same motion, different appearance/context**, final-embedding retrieval, and V-JEPA-versus-VideoMAE/TimeSformer. It does not provide layerwise probes or latent causal interventions. Its synthetic videos are synchronized image-to-video generations from paired start frames, not repeated renders of one simulator state; consequently, the paper's “exact same motion dynamics” is a generator-level claim, not a ground-truth guarantee of identical 3D trajectories, contacts, camera, and physics. Dynamic-object and viewpoint changes also exceed a strict nuisance-appearance definition. Exact renderer-state counterfactuals remain a meaningful improvement.

### 2. A near-direct broad JEPA robustness benchmark already exists

[Latent Video Prediction Learns Better World Models, arXiv:2605.15618](https://arxiv.org/abs/2605.15618) compares ViT-L V-JEPA 2.1, V-JEPA 2, VideoPrism, and VideoMAE-v2 on SSv2. Its five axes are feature discriminability, six ImageNet-C corruptions, pretend-action discrimination, three occlusion families, and temporal perturbations. It reports V-JEPA 2.1 leading five of six corruptions and retaining 46.1% top-1 under severe patch dropout, while VideoPrism retains cosine similarity above 0.98 but falls to 2.7% accuracy.

What it leaves open:

- no layerwise analysis;
- no image I-JEPA track;
- no paired renderer-level intervention that certifies identical scene state and physics;
- no direct separation of appearance information being present from being functionally used;
- no released code link in the source;
- objective attribution remains confounded by pretraining data, schedule, model implementation, preprocessing, and probe tuning.

The paper calls the models capacity-matched, but its appendix acknowledges unmatched pretraining corpora and schedules. It selects a different best attentive-probe configuration per encoder using clean validation accuracy, while clean SSv2 accuracies differ substantially. Its separate fine-tuning comparison also mixes ViT-L V-JEPA with ViT-B VideoMAE and model-native frame counts/resolutions. These facts weaken causal claims about the training objective and define mistakes the proposed benchmark should avoid.

### 3. The layerwise and causal templates are occupied

[Joseph et al., *Interpreting Physics in Video World Models*, arXiv:2602.07050](https://arxiv.org/abs/2602.07050) already combines layerwise linear/attentive probing, patch-level decoding, probe-subspace geometry, attention ablation, and activation steering on V-JEPA 2 and VideoMAE-v2. It identifies a one-third-depth Physics Emergence Zone.

[Kowal et al., *A Deeper Dive Into What Deep Spatiotemporal Networks Encode*, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Kowal_A_Deeper_Dive_Into_What_Deep_Spatiotemporal_Networks_Encode_Quantifying_CVPR_2022_paper.html) already quantify static versus dynamic information layerwise and channelwise using paired shuffled and stylized videos, and perform top-ranked unit ablations. Their supported architectures include TimeSformer, MViT, SlowFast, X3D, I3D, and C2D.

Simply changing Joseph et al.'s target from physics to appearance, or rerunning Kowal et al. on V-JEPA, is likely too incremental. A viable paper needs exact paired counterfactuals, factor-specific appearance labels, a rigorous distinction between decodability and functional dependence, and native downstream/predictor effects.

Joseph et al.'s results also warn against a JEPA-specific zone claim: VideoMAE-v2-G shows a related one-third-depth emergence transition. Their steering changes a separately trained held-out direction readout, not native physical prediction or planning. Their attention ablation gives stronger functional evidence but is broad rather than appearance-specific. Full WhiteHole-focused summary: [local summary](../paper_summaries/summary_2602.07050_interpreting-physics-video-world-models/summary.md).

A workshop paper titled [*Causal State Variables in V-JEPA 2 Latents*](https://openreview.net/forum?id=CC5xKstwmf) also claims a causal intervention pipeline, early-layer motion encoding, and subspace-portability metrics on V-JEPA 2. Its PDF was blocked by OpenReview verification during this audit, so those details are not independently verified; nevertheless, a “first causal intervention in V-JEPA 2” claim is unsafe.

### 4. Controlled invariance and appearance-bias benchmarks predate JEPA

- [PUG](https://proceedings.neurips.cc/paper_files/paper/2023/file/8d352fd0f07fde4a74f9476603b3773b-Paper-Datasets_and_Benchmarks.pdf) provides photorealistic, controlled image factors. PUG-ImageNet has 88,328 images, 151 ImageNet classes, 724 assets, 64 environments, texture, light, size, camera, and object-orientation factors.
- [3DIEBench/SIE](https://proceedings.mlr.press/v202/garrido23b.html) provides 2.5M controlled 3D renders over 55 classes and explicitly studies invariant versus equivariant representations.
- [Demystifying Contrastive SSL](https://arxiv.org/abs/2007.13916) measures viewpoint, illumination/color, and occlusion invariance and shows that objective, augmentations, and dataset bias all matter.
- [HAT](https://proceedings.neurips.cc/paper_files/paper/2022/file/ff52407b80dde0f0f45814db2738464c-Paper-Datasets_and_Benchmarks.pdf) evaluates 74 action models using background-only, human-only, and action-swap videos generated through segmentation and video inpainting.
- [SCUBA/SCUFO](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Mitigating_and_Evaluating_Static_Bias_of_Action_Representations_in_the_ICCV_2023_paper.pdf) explicitly tests background and foreground static bias in action recognition.
- [Mini Kinetics-C/Mini SSv2-C](https://arxiv.org/abs/2110.06513) and [Kinetics400-P/SSv2-P](https://openaccess.thecvf.com/content/CVPR2023/papers/Schiappa_A_Large-Scale_Robustness_Analysis_of_Video_Action_Recognition_Models_CVPR_2023_paper.pdf) already cover broad video corruption and perturbation robustness.

Therefore, neither controlled factors, video corruption, appearance/motion bias, nor layerwise probing is independently novel. The intersection **paired physical-state-preserving counterfactuals + frozen JEPA internals + functional dependence** is the plausible gap.

### 5. Image-side appearance probing is also occupied at the output layer

[Substance or Style: What Does Your Image Embedding Know?](https://arxiv.org/abs/2307.05610) trains probes on frozen MAE, CAN, SimCLR, CLIP, ALIGN, and supervised image embeddings to identify dozens of color, quality, corruption, overlay, geometric, and style-transfer manipulations. It directly establishes that semantic embeddings can retain rich non-semantic transformation information. It does not study I-JEPA, full layer trajectories, paired content retention, or functional use by subsequent frozen blocks.

[Image World Models](https://arxiv.org/abs/2403.00504) is the closest JEPA-specific image precedent: it explicitly trains latent predictors for color jitter, grayscale, blur, and solarization and studies invariant versus equivariant representations. It changes the training objective and model, however, rather than auditing released frozen I-JEPA checkpoints.

[What Do Self-Supervised Vision Transformers Learn?](https://openreview.net/forum?id=azCKuYyS74) and [Objectives Matter](https://arxiv.org/abs/2304.13089) already compare layerwise organization of joint-embedding/contrastive and masked-image objectives, including shape/texture, frequency, locality, and attention differences. Therefore, “JEPA is more shape-biased than MAE” or “latent and pixel objectives organize layers differently” is not enough by itself.

[Support×Operation Factorization](https://arxiv.org/abs/2608.06174) is a recent close benchmark precedent: it measures paired patch-token changes under controlled support × operation interventions in frozen DINOv3 and SigLIP2. It studies compositional readout at the final token map, not appearance invariance, content retention, complete layer sweeps, or causal propagation through later blocks.

### 6. A generic intermediate-layer pattern has an alternative explanation

[Odonnat et al., arXiv:2603.05280](https://arxiv.org/abs/2603.05280) show that stronger pretraining-to-downstream distribution shift makes later ViT layers less useful for linear OOD probing. They also show that probing standard block outputs can be suboptimal: intermediate FFN activations work best under strong shift, while normalized pre-FFN representations are safer under weak shift.

Consequently, declining late-layer appearance or content probe accuracy does not establish “discard.” It can be caused by generic OOD specialization, probe location, pooling, or changed linear accessibility. The benchmark must probe matched module locations and include nonappearance OOD controls.

## Competitor map

| Category | Work | What it already establishes | Remaining opening |
|---|---|---|---|
| Direct behavioral | [SimMotion](https://arxiv.org/abs/2602.09146) | Same-motion retrieval across five visual-factor families with V-JEPA 2, VideoMAE, TimeSformer, and others | Exact simulator-state guarantees; layers; causal use; native predictor |
| Direct behavioral | [Latent Video Prediction Learns Better World Models](https://arxiv.org/abs/2605.15618) | V-JEPA 2/2.1 versus VideoMAE-v2/VideoPrism under corruption, occlusion, contact, and temporal shifts | Factorized appearance counterfactuals; layers; causal use |
| Direct representation analysis | [Kowal et al.](https://openaccess.thecvf.com/content/CVPR2022/html/Kowal_A_Deeper_Dive_Into_What_Deep_Spatiotemporal_Networks_Encode_Quantifying_CVPR_2022_paper.html) | Layerwise/channelwise static-dynamic information and ablation | Modern JEPAs; exact factors; predictor behavior |
| Direct interpretability | [Interpreting Physics in Video World Models](https://arxiv.org/abs/2602.07050) | Layerwise probes, subspaces, steering, and attention ablation in V-JEPA 2/VideoMAE-v2 | Appearance held apart from fixed physics; representation-versus-use |
| Adjacent disentanglement | [DisMo](https://compvis.github.io/DisMo/) | Learns content-invariant motion representations and compares against V-JEPA on SSv2/Jester | Audits a new trained model, not frozen JEPA internals |
| Adjacent static-bias evaluation | [HAT](https://github.com/princetonvisualai/HAT), [SCUBA/SCUFO](https://github.com/lihaoxin05/StillMix), [AFD101](https://github.com/f-ilic/AppearanceFreeActionRecognition) | Background/foreground interventions and an appearance-free action endpoint | Broad factorization, layers, JEPA causal mechanism |
| Adjacent robustness | [Video perturbation benchmark](https://openaccess.thecvf.com/content/CVPR2023/html/Schiappa_A_Large-Scale_Robustness_Analysis_of_Video_Action_Recognition_Models_CVPR_2023_paper.html), [Video-C](https://github.com/Newbeeyoung/Video-Corruption-Robustness) | 90 perturbations and standard corruption suites | Exact equivalent-state counterfactuals and mechanism |

## Useful released datasets and generators

| Resource | Best use here | Important limitation |
|---|---|---|
| [SimMotion-Synthetic](https://huggingface.co/datasets/Shuberman/SimMotion-Synthetic) / [Real](https://huggingface.co/datasets/Shuberman/SimMotion-Real) | Mandatory behavioral baseline and direct prior benchmark | Generated synchronization is not simulator ground truth; only 250 synthetic and 40 real triplets |
| [Kubric / MOVi](https://github.com/google-research/kubric/tree/main/challenges/movi) | Generate repeated renders from one physics rollout with masks, depth, flow, and trajectories | Existing MOVi releases are not organized as exact same-rollout appearance pairs; custom generation is needed |
| [Temporal Shape](https://github.com/sofiabroome/temporal-shape-dataset) and [modified Diving48](https://github.com/sofiabroome/cross-dataset-generalization) | Lightweight temporal/domain-shift controls and texture/shape tests | Small or task-specific; trained-from-scratch predecessor protocol differs from frozen foundation models |
| [SCUBA/SCUFO/ConflFG](https://github.com/lihaoxin05/StillMix) | Background, foreground-static, and conflicting-cue evaluation | Compositing artifacts and incomplete pure appearance coverage |
| [HAT](https://github.com/princetonvisualai/HAT) | Human-only, background-only, and action-swap Kinetics/UCF variants | Segmentation/inpainting may alter evidence |
| [AFD101](https://github.com/f-ilic/AppearanceFreeActionRecognition) | Dynamic-only endpoint where no single frame is class-discriminative | Removes appearance cues rather than varying them naturally |
| [ARAS-104](https://github.com/kennymckormick/ARAS-Dataset) and [Mimetics](https://arxiv.org/abs/1912.07249) | Natural background/context domain shift | Unpaired; action execution and other content also change |
| [Mini Kinetics-C / Mini SSv2-C](https://github.com/Newbeeyoung/Video-Corruption-Robustness) and [HMDB/UCF/Kinetics/SSv2-P](https://openaccess.thecvf.com/content/CVPR2023/html/Schiappa_A_Large-Scale_Robustness_Analysis_of_Video_Action_Recognition_Models_CVPR_2023_paper.html) | Standard corruption baselines | Sensor degradation is not equivalent to appearance-only intervention |

## Recommended benchmark design

### Operational variables

For every scene or trajectory, define:

- **State/content variables `Y`:** object identity/shape, pose, depth, trajectory, velocity, direction, contact/collision state, or action.
- **Visual factors `A`:** only factors independently randomized by the renderer or post-processing pipeline.
- **Fixed variables:** mesh geometry, object pose sequence, camera intrinsics/extrinsics, frame times, physics engine state, contacts, mass, friction, restitution, and action labels.

Call a factor “nuisance” only relative to an explicit `Y`. Material appearance is not automatically nuisance: in real data it can indicate friction, deformability, or mass. The benchmark should say “appearance intervention with respect to target Y,” not assert a universal appearance/content partition.

### Core data: paired counterfactual renders

Use a factorial renderer design in which every latent scene trajectory is rendered under multiple independently assigned appearances. A practical video source is Kubric/Blender with physics generated once and rendered repeatedly.

**Primary factors:**

1. object albedo/texture;
2. shading material, with physical friction/restitution held separately fixed;
3. illumination color and intensity;
4. background/environment appearance;
5. global photometric sensor mapping such as white balance, gamma, or channel response.

**Separate diagnostic axes:**

- **Information-destroying sensor degradation:** blur, noise, compression, occlusion. These test graceful degradation, not invariance to an equivalent observation.
- **Geometry/observation changes:** crop, shear, image rotation, camera viewpoint, focal length. These must not be labeled appearance-only.
- **Temporal observation changes:** per-frame jitter, flicker, frame-rate change, or moving camera. These change observed dynamics and belong in a temporal/camera suite.

Apply a visual transform identically across the clip unless temporal variation is the variable under study. Independent framewise stylization introduces artificial motion.

For images, use PUG-ImageNet first: texture, environment, and light are controlled appearance factors; camera orientation, object orientation, and size are explicit geometry controls. This is cleaner and cheaper than claiming that arbitrary style transfer preserves semantics.

### Natural-video external validity

Use HAT action swaps and SCUBA/SCUFO only as supporting evidence. They preserve real foreground motion better than generic stylization, but segmentation boundaries, inpainting, alignment, and compositing artifacts can become model shortcuts. Report artifact-only controls, such as compositing a foreground back onto its own reconstructed background and swapping between backgrounds of the same scene class.

### Do not make neural style transfer a core condition

Style-transfer output is not ground-truth appearance-only. It can change silhouettes, edge locations, local geometry, object identity, visibility, and temporal consistency. If retained as a stress test:

- use a temporally consistent method;
- verify optical flow, segmentation masks, keypoints, and depth boundaries before/after;
- include human semantic-consistency checks;
- label it an uncontrolled naturalistic shift, not a causal factor intervention.

## Measurements that answer distinct questions

At each encoder layer and pre-specified module location:

1. **Appearance accessibility:** cross-validated linear prediction of `A`.
2. **Content/physics accessibility:** prediction of `Y` from the same representation.
3. **Source-readout retention:** train a probe on one canonical appearance and evaluate unchanged across paired appearances.
4. **Target-oracle accessibility:** train/evaluate a probe within each shifted appearance.
5. **Cross-appearance retrieval:** retrieve the same trajectory/state across appearance variants against hard negatives with similar appearance but different state.
6. **Native predictive behavior:** where a released JEPA predictor is available, compare masked/future latent predictions across paired renders. Keep this separate from classification probes because predictor loss is not commensurable with pixel reconstruction loss.

The source-versus-oracle comparison is essential:

- source probe drops, oracle remains high → content is present but represented in shifted coordinates;
- both drop → the representation lost accessible content;
- neither drops while appearance remains decodable → coexistence or decoupling, not deletion.

Do not use clean/shift cosine similarity as the main score. arXiv:2605.15618 directly shows cosine stability can coexist with near-chance classification. Normalize paired displacement by between-trajectory margins and always pair geometry metrics with usable readouts.

## Probe protocol required for a credible layerwise claim

- Split by latent scene/trajectory, with every appearance variant of a scene in exactly one split. Frame- or render-level random splits leak content.
- Randomize appearance factors independently of `Y`; use a crossed design so no style label predicts class, trajectory, renderer seed, or split.
- Pre-register pooling and module locations. At minimum report block residual output and pre-FFN normalized/activation outputs because arXiv:2603.05280 shows module choice changes OOD conclusions.
- Use fixed-capacity linear probes with nested validation, equal search budgets, multiple seeds, label-shuffle controls, and bootstrap confidence intervals grouped by trajectory.
- Report both global pooled and patch-preserving readouts. Mean pooling can erase local signals; a large attentive probe can solve the task itself.
- Compare within-model layer trajectories using relative depth. Raw probe scores are not directly comparable across representation width, normalization, token count, and frame count.
- Calibrate clean-task difficulty and report absolute and relative performance to avoid floor/ceiling artifacts.

## Causal intervention standard

The paired renderer intervention is already the cleanest causal experiment: it changes `A` while holding the scene state fixed. Make this the main causal evidence.

A latent “appearance-subspace swap” should be optional until it passes all of the following:

- **Identification:** estimate the subspace from factorial appearance contrasts, not from naturally correlated labels.
- **Completeness:** linear and nonlinear held-out probes can no longer recover the targeted factor after erasure or correctly recover the donor value after replacement.
- **Selectivity:** independent probes for shape, pose, motion, identity, and physics remain intact.
- **Functional readout:** measure an existing downstream head or frozen predictor not used to construct the intervention.
- **Negative controls:** equal-rank random subspace, orthogonal complement, content subspace, and activation-norm-matched noise.
- **Dose response:** vary intervention rank/strength rather than selecting one favorable point.
- **Distribution validity:** measure whether edited states remain near the natural activation manifold.

This standard follows the core warning from [Amnesic Probing](https://aclanthology.org/2021.tacl-1.10/): decodability does not imply behavioral use. Probe-derived erasure can also destroy unrelated task information or fail to remove the intended concept ([Kumar et al. 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/725f5e8036cc08adeba4a7c3bcbc6f2c-Paper-Conference.pdf)). Recent causal-probing evaluation formalizes the completeness/selectivity trade-off ([Canby et al. 2025](https://aclanthology.org/2025.ijcnlp-long.47/)).

## Minimum convincing experiment set

### Models

**Image:** I-JEPA ViT-H/14, a closest-capacity MAE control, and DINOv2 as a non-pixel, non-JEPA SSL control.  
**Video:** V-JEPA 2 or 2.1 ViT-L, VideoMAE-v2 ViT-L, and one non-JEPA SSL video encoder if compute allows.

Keep official preprocessing native to each checkpoint, but evaluate identical underlying renders and document resolution, frame count, cadence, crop, pooling, and token selection. The public-checkpoint result must be described as a model-family comparison.

### Data and factors

- **Image:** PUG-ImageNet; texture, environment, and lighting as appearance; camera/object orientation as geometry controls.
- **Video:** at least a multi-object Kubric collision/interaction set, not only a moving ball. Re-render each trajectory under object texture/material, lighting, background, and global photometric factors.
- **Supporting real-video test:** HAT action-swap or SCUBA, with reconstruction/artifact controls.

A credible screen needs enough independent trajectories for grouped splits and uncertainty estimates. Repeated renderings do not increase the effective sample size for physics labels. The project's current 100-clip, five-class Kinetics-mini split is suitable only for pipeline debugging; its previous 88% clean and shifted result supplies neither power nor a useful robustness gap.

### Required analyses

1. Layerwise `A` and `Y` probes.
2. Frozen clean-source probe versus shifted evaluation and per-shift oracle probes.
3. Paired cross-appearance retrieval and normalized geometry drift.
4. Factor composition and held-out severity tests, not only seen single shifts.
5. One functional validation through the frozen V-JEPA predictor or an independently trained source downstream head.
6. Three or more probe seeds plus trajectory-level bootstrap intervals.

An internal subspace intervention is not required for the first convincing paper. Add it only after the paired behavioral and layerwise results establish a phenomenon worth mechanistically testing.

## Claims the evidence would and would not support

**Supportable from public checkpoints:**

- where specific appearance factors are linearly accessible;
- whether content/physics remains accessible under paired appearance interventions;
- whether clean readouts transfer or require shifted coordinates;
- whether model families differ in these profiles.

**Not supportable without matched retraining or stronger behavior:**

- latent prediction causes appearance invariance;
- JEPA learns appearance/physics disentanglement in general;
- a probe direction is the model's appearance mechanism;
- robustness implies physical understanding;
- a classification/readout result implies planning robustness.

## Concrete recommendation

Build the paper around **counterfactual appearance sensitivity and functional decoupling**, with “Appearance-Discard Zone” retained only as a possible empirical outcome. Use PUG for the image half and paired Kubric re-rendering for the video half. Treat broad corruption benchmarks as related work or one external-validity table. Do not lead with style transfer, raw cosine similarity, or a causal objective claim.

A strong result would be one of these:

- appearance becomes less decodable while physics/content remains available;
- appearance remains decodable but ceases to perturb source readouts or native prediction;
- no such transition exists, challenging the common JEPA abstraction intuition;
- JEPA and pixel-reconstruction families differ consistently, stated as a checkpoint-family observation unless replicated in a matched controlled pretraining experiment.

## Source and repository audit

The operational inventory of locally available repositories, checkpoints, generators, public origins, and reuse limitations is maintained in [Reusable JEPA and world-model resource inventory](reusable_jepa_world_model_resources.md).

Full TeX trees inspected:

- arXiv:2602.07050, `/tmp/arxiv-src/2602.07050/`;
- arXiv:2602.09146, `/tmp/arxiv-src/2602.09146/`;
- arXiv:2605.15618, `/tmp/arxiv-src/2605.15618/`;
- arXiv:2603.05280, `/tmp/arxiv-src/2603.05280/`;
- arXiv:2112.12175, `/tmp/arxiv-src/2112.12175/`.

Repositories inspected without modifying the WhiteHole checkout:

- [saarhub/semantic-moments](https://github.com/saarhub/semantic-moments), commit `eb4ec98f75d287db75acaeeed72840a7f4aeb217`;
- [YorkUCVIL/Static-Dynamic-Interpretability](https://github.com/YorkUCVIL/Static-Dynamic-Interpretability), commit `f552b31db1f34b622decc39b72206a55d800948f`;
- [princetonvisualai/HAT](https://github.com/princetonvisualai/HAT), commit `bda849b341e5f0200b311a112e3500582a30e02f`;
- [lihaoxin05/StillMix](https://github.com/lihaoxin05/StillMix), commit `e5a9d5a0a0f666e5254d4850db33f5b3daf8230f`;
- [facebookresearch/SIE](https://github.com/facebookresearch/SIE), commit `54541cfa40780aa3e502e3bbb52b30aae672b487`.

The OpenReview PDF for *Causal State Variables in V-JEPA 2 Latents* could not be fetched because both the API and PDF endpoint returned browser-verification 403 responses. Only its OpenReview title and the author's public description were used, and no detailed empirical claim from that paper is treated as verified.
