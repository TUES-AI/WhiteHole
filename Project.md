# WhiteHole

## Status

WhiteHole is early exploratory research on frozen JEPA representations and latent world models under changes in visual observation.

The previous adapter-development track produced useful negative and calibration evidence, but it has not produced a general adaptation method. The team is now considering a research pivot toward a **counterfactual analysis and benchmark of appearance sensitivity versus physical selectivity in frozen JEPAs**.

**This pivot is provisional, not a final team decision.** The current document fuses the Toni, Alex, and Vasi proposals so collaborators and their agents share one working context. The team will discuss the scope before experiments resume, after which this document must be updated again.

The three source proposals are:

- [Vasi: Appearance-Discard Zones in JEPA Representations](docs/research_directions/1508-appearance_discard_zones_in_JEPA_representations-vasi.md)
- [Alex: Appearance-Invariant Physical Surprise](docs/research_directions/1808-jepa_appearance_invariant_physical_surprise_research_direction-alex.md)
- [Toni: Counterfactual Appearance Sensitivity in Frozen JEPAs](docs/research_directions/1808-counterfactual_appearance_sensitivity_in_frozen_jepas-toni.md)

## Provisional research question

> **Under exact paired counterfactual observations that preserve physical state and dynamics while changing visual appearance, where across frozen JEPA depth does appearance remain accessible, and does retained appearance alter physical-state readouts, native prediction, or violation-of-expectation surprise?**

An informal version is:

> **When nothing physical changes, what changes inside a visual world model—and does that change what the model believes about physics?**

The central distinction is between:

1. **appearance accessibility:** appearance can be decoded from an activation;
2. **physical-state accessibility:** content, trajectory, or physical state remains recoverable across appearances;
3. **functional dependence:** appearance changes native prediction, physical-surprise margins, frozen downstream behavior, action ranking, or planning.

Appearance does not need to disappear for a useful abstraction to exist. It may remain decodable while becoming functionally decoupled from physical computation.

## Fused working direction

### Pillar 1: exact counterfactual observations

Generate physics once and rerender the same simulator-state sequence under independently controlled appearances.

For state or trajectory \(S\) and appearance \(A\):

```text
anchor:   (S,  A)
positive: (S,  A')   same physics, different appearance
negative: (S', A)    different physics, same appearance
```

Keep fixed whenever an intervention is labeled appearance-only:

```text
geometry and object identity
pose, velocity, acceleration, contacts, and collisions
camera intrinsics/extrinsics
frame times and cadence
mass, friction, restitution, and actions
trajectory and simulator state
```

Vary independently:

```text
object albedo and texture
visual material while physical material parameters remain fixed
illumination color and intensity
background/environment appearance
global camera response such as white balance or gamma
```

Do not combine appearance with other intervention families:

- **sensor degradation:** noise, blur, compression, pixelation, occlusion;
- **observation geometry:** viewpoint, crop, rotation, shear, moving camera;
- **temporal observation:** frame rate, drops, flicker, reversal, shuffle.

These can be diagnostic suites but require separate claims.

### Pillar 2: layerwise accessibility and transfer

At every frozen encoder layer, for pooled and patch-preserving representations, measure:

1. appearance-factor probes;
2. content/physical-state probes;
3. a source-trained probe evaluated unchanged across appearances;
4. an appearance-specific oracle probe;
5. same-state cross-appearance retrieval with same-appearance/different-state hard negatives;
6. paired appearance displacement relative to physical-state displacement.

The source-versus-oracle comparison distinguishes:

- source probe fails while oracle succeeds: state information remains but moved coordinates;
- both fail: accessible state information was lost;
- neither fails while appearance remains decodable: coexistence or functional decoupling.

“Appearance-Discard Zone” is a possible empirical outcome, not the assumed framing. Neutral terms are **appearance sensitivity**, **appearance accessibility**, and **functional decoupling**.

### Pillar 3: integrity of physical surprise

For a matched physically plausible video \(V^+\) and impossible video \(V^-\), cross physical condition with the same appearance interventions. If \(S(V)\) is native latent prediction surprise, study the margin

\[
M_a = S(T_a(V^-)) - S(T_a(V^+)).
\]

The main questions are:

- does the correct plausible/impossible ordering survive appearance changes?
- does appearance add a common surprise offset while preserving the margin?
- does appearance suppress or amplify the violation margin?
- does it create false temporal peaks or move the peak away from the violation frame?
- are effects specific to physical concepts such as continuity, solidity, support, collision, or permanence?

Initially report a small interpretable suite rather than many branded metrics:

- pairwise physical ordering;
- margin change relative to the original appearance;
- appearance-induced surprise variation relative to the physics margin;
- single-video AUROC by appearance;
- temporal violation-localization error.

### Pillar 4: native functional dependence

Probe accuracy and representation similarity do not establish functional use. Where models permit, evaluate:

- native JEPA masked/future prediction;
- independently trained frozen source readouts;
- action-conditioned rollout separation;
- action-ranking agreement;
- MPC or closed-loop planning.

Physical surprise is the preferred sharp functional endpoint for frozen V-JEPA-style models. Planning is a stronger endpoint for action-conditioned LeWorldModel, DINO-WM, or PLDM-style systems. The final project may select only one of these to keep the paper focused.

## Candidate models

### Image

- I-JEPA;
- MAE as a reconstruction control;
- DINOv2 or DINOv3 as non-JEPA self-supervised controls.

### Video

- V-JEPA 2 or 2.1;
- VideoMAE-v2;
- optionally VideoPrism or DINO frame features.

### Action-conditioned functional validation

- LeWorldModel;
- DINO-WM or PLDM-style planning environments.

Public-checkpoint comparisons support conclusions about those model families. They do not show that latent prediction caused an effect because architecture, data, scale, augmentations, training schedule, and compute remain confounded. Objective-causality claims require matched retraining.

## Candidate datasets

- **CRONOS:** an existing Unreal Engine pilot with collisions, falls, occlusion, and controlled appearance/scene/object/viewpoint variants. Its exact pairing and appearance diversity must be audited before adoption; it evaluates video generators in the original paper, not frozen JEPAs.
- **IntPhys / IntPhys 2:** plausible/impossible physics stimuli for the surprise track. Appearance interventions must be temporally coherent and crossed identically with each physical condition.
- **PUG-ImageNet:** controlled image texture, environment, and lighting. Orientation, camera, and size are geometry controls.
- **Custom repeated rendering:** likely required for the strongest video claim if existing datasets do not provide enough exact same-state appearance factors.
- **LeWorldModel environments:** secondary functional validation through native prediction and planning.

Repeated renders increase appearance observations but not the effective number of independent trajectories for physical labels. Splits and bootstrap units must be grouped by base scene or trajectory.

## Closest work and novelty constraints

- [Interpreting Physics in Video World Models](https://arxiv.org/abs/2602.07050) already performs layerwise physics probing, subspace analysis, attention suppression, and steering in V-JEPA 2 and VideoMAE-v2.
- [SemanticMoments / SimMotion](https://arxiv.org/abs/2602.09146) already tests same-motion retrieval under changed subject, scene, style, and viewpoint.
- [Latent Video Prediction Learns Better World Models](https://arxiv.org/abs/2605.15618) already benchmarks V-JEPA robustness to corruption, occlusion, contact cues, and temporal order.
- [PhyLatent](https://arxiv.org/abs/2608.05720) already defines physical invariance collapse by comparing final-latent appearance displacement with real state displacement in LeWorldModel and adds supervised invariance training.
- [CRONOS](https://arxiv.org/abs/2605.23699) already evaluates counterfactual physical consistency of video generators under visual interventions.
- [Joint Embedding Predictive Architectures Focus on Slow Features](https://arxiv.org/abs/2211.10831) shows that predictive objectives can prefer stable distractors rather than discard them.
- [Learning Invariant Visual Representations for Planning with Joint-Embedding Predictive World Models](https://arxiv.org/abs/2602.18639) already improves DINO-WM planning under background changes through a learned bisimulation representation.
- [Image World Models](https://arxiv.org/abs/2403.00504) studies invariant/equivariant representations under transformation-conditioned image prediction.
- [Substance or Style](https://arxiv.org/abs/2307.05610) probes non-semantic transformation information in frozen image embeddings.

Alex’s proposal additionally identifies the very recent ACPC and UniJEPA papers as potential novelty constraints. Their complete source and claims must be independently audited before the team finalizes the scope.

The plausible remaining opening is the intersection:

```text
exact same-state appearance counterfactuals
+ frozen image/video JEPA foundation models
+ complete layer trajectories
+ accessibility versus native functional dependence
+ physics-surprise integrity
```

The full literature review is in [the senior-review report](docs/research_reports/jepa_visual_shift_benchmark_review.md), with reusable code, checkpoints, datasets, and origins in [the resource inventory](docs/research_reports/reusable_jepa_world_model_resources.md).

## Working hypotheses and valid outcomes

1. **Appearance becomes inaccessible while physical state remains accessible.** This would support a genuine appearance-discard transition.
2. **Appearance remains decodable but stops altering physical readouts or surprise.** This would support functional decoupling without deletion.
3. **Source readouts fail while appearance-specific oracles succeed.** Physical information survived in appearance-dependent coordinates.
4. **Appearance changes the magnitude or timing of physical surprise.** Existing violation-of-expectation evidence is appearance-contingent.
5. **No localized transition exists.** This would challenge the intuitive claim that latent prediction creates a clean abstraction boundary.
6. **JEPA and reconstruction models behave similarly.** Public-checkpoint evidence would not support a JEPA-specific mechanism.

Any of these can be scientifically useful if supported by exact controls and uncertainty estimates.

## Existing WhiteHole evidence

The completed adapter experiments remain relevant as methodological warnings, not evidence for the new benchmark.

### Frozen I-JEPA and V-JEPA adapter screen

Committed code and artifacts are at commit `a91162d` and `results/11-17-08-jepa-visual-adapters/`.

- I-JEPA clean probe accuracy was 95.0%; fixed RBG, affine, and composed shifts scored 92.7%, 90.4%, and 78.3%.
- Approximately 70–98K adapters passed paired pixel recovery only for RBG; target-only masked-I-JEPA did not improve downstream RBG accuracy.
- The geometry failures reject only the tested architecture, objective, source-retention formulation, data, and optimization budget.
- The V-JEPA Kinetics-mini screen scored 88% on both clean and RBG and therefore exposed no downstream gap to repair.
- Lower pixel or JEPA prediction loss did not reliably rank downstream recovery.

The old “capacity gate” is more accurately **paired recovery with source retention**. It simultaneously tested correction, clean-input preservation, unconditional routing, architecture, data, and optimization; it was not a pure neural-capacity test.

### Action-conditioned Reacher adaptation

The historical Reacher experiments established that:

- dynamics or representation loss can improve while control collapses;
- STN-only adaptation can outperform broader encoder updates;
- semantic dynamic backgrounds are not repaired by simple affine adaptation;
- a supervised coordinate-aware U-Net provides an upper bound, not deployment-available adaptation;
- an exact projective camera canonicalization oracle strongly recovers dynamic-camera control, while learned small adapters do not.

These results reinforce the new direction’s requirement to separate representation displacement, prediction, and native behavior.

Full experiment records remain append-only in `Science_log.md` and `results/`.

## Evaluation principles

- Split by base scene or trajectory; keep every appearance variant in one split.
- Randomize appearance independently of physical labels.
- Use fixed-capacity probes with equal search budgets and multiple seeds.
- Report pooled and patch-preserving representations.
- Compare matched module locations and relative depth across models.
- Pair invariance with physics discrimination so collapse cannot appear successful.
- Use source-trained and appearance-specific oracle readouts.
- Treat cosine similarity, CKA, probe accuracy, native prediction, and planning as different measurements.
- Label every visual factor as nuisance, physics-relevant, or ambiguous relative to the target variable.
- Apply transformations consistently across time unless temporal appearance is explicitly studied.
- Do not infer objective causality from unmatched public checkpoints.
- Record exact commands, seeds, splits, budgets, checkpoints, transformations, and artifacts for every executed experiment.

## Decision required before execution

No new benchmark or model run should begin until the team decides:

1. whether physical surprise is the primary endpoint or one functional endpoint within a broader layerwise audit;
2. whether the first dataset is CRONOS, IntPhys, or a new exact renderer;
3. whether both image and video tracks are required for the first paper;
4. whether the contribution is diagnostic/benchmark-only or includes a small calibration method;
5. which recent papers, particularly ACPC and UniJEPA, materially constrain novelty.

After the team discussion, update this document before beginning execution. Do not resume broad adapter tuning unless the team explicitly returns to adaptation as the primary contribution.
