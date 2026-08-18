# Do JEPA World Models Understand Physics Beyond Appearance?
## Counterfactual Evaluation of Appearance-Invariant Physical Surprise

**Research direction / working proposal**  
**Status:** Concept proposal, literature checked through **18 August 2026**

---

## 1. Core idea

Joint-Embedding Predictive Architectures (JEPAs) are designed to predict in a learned representation space rather than directly reconstructing pixels. This creates an appealing hypothesis: a JEPA world model may learn to represent *physically meaningful structure* while becoming relatively insensitive to visually salient but physically irrelevant details.

Recent work has shown that V-JEPA can distinguish physically plausible from implausible videos using a **surprise signal** derived from latent-space prediction error. However, prediction error alone does not tell us *why* the model is surprised.

A model may be surprised because:

1. an object violates a physical rule;
2. the object's color or texture is unusual;
3. the lighting or background is out of distribution;
4. some interaction between appearance and dynamics changes the model's prediction difficulty.

This motivates the central question:

> **If the physical state and dynamics of a scene remain unchanged while its visual appearance is counterfactually modified, does a JEPA preserve its judgment of physical plausibility?**

The goal is not merely to measure ordinary augmentation robustness. The goal is to determine whether **physical surprise is causally tied to physical violations rather than nuisance appearance variables**.

---

## 2. Proposed paper framing

### Possible title

**Do JEPA World Models Understand Physics Beyond Appearance? Counterfactual Evaluation of Appearance-Invariant Physical Surprise**

Alternative titles:

- **Appearance-Invariant Surprise as a Test of Physical Understanding in Latent World Models**
- **What Does JEPA Surprise Measure? Disentangling Physical Violations from Visual Appearance**
- **Counterfactual Physical Surprise: Testing Whether Latent World Models Abstract Away Appearance**
- **Beyond Violation of Expectation: Measuring Appearance Leakage in Predictive World Models**

### One-sentence thesis

> A model should not be considered to possess a robust intuitive physical world model merely because impossible events produce larger prediction error; the *physical-surprise signal itself* should remain stable under interventions on visually salient but physically irrelevant variables.

---

## 3. Why this matters

The violation-of-expectation paradigm evaluates physical understanding by comparing a model's response to possible and impossible events. Garrido et al. showed that V-JEPA's latent prediction error can act as a quantitative surprise signal and reported strong zero-shot performance on several intuitive-physics benchmarks.

For a video \(V\), their formulation measures surprise at time \(t\) using the distance between the predicted future representation and the encoded observed future:

\[
S_t =
\left\|
p_\phi(f_\theta(V_{t:t+C}))
-
g_\psi(V_{t:t+C+M})
\right\|_1 .
\]

Video-level surprise can then be aggregated using, for example,

\[
S_{\mathrm{avg}}(V)
=
\frac{1}{T}\sum_t S_t
\]

or

\[
S_{\mathrm{max}}(V)
=
\max_t S_t.
\]

A relative physical-surprise score for a matched possible/impossible pair is

\[
\Delta_{\mathrm{phys}}
=
S(V^-)-S(V^+),
\]

where \(V^+\) is physically plausible and \(V^-\) contains a physical violation.

If the model has captured the relevant physical expectation, we expect

\[
\Delta_{\mathrm{phys}} > 0.
\]

However, this measurement does not by itself establish that \(\Delta_{\mathrm{phys}}\) is invariant to physically irrelevant appearance changes.

A robust physical representation should ideally satisfy:

\[
\Delta_{\mathrm{phys}}(T_a(V))
\approx
\Delta_{\mathrm{phys}}(V)
\]

for transformations \(T_a\) that modify appearance but preserve the underlying physical state trajectory.

---

## 4. Research question

### Main research question

> **To what extent is physical surprise in JEPA-style predictive world models invariant to counterfactual changes in visual appearance that preserve the underlying physical dynamics?**

### Secondary questions

1. Which appearance factors cause the largest distortion of JEPA surprise?
2. Does appearance primarily add a common surprise offset, or does it change the *relative margin* between possible and impossible events?
3. Are failures localized to the encoder representation or amplified by the predictor?
4. Does robustness improve with model scale, pretraining scale, or newer JEPA objectives?
5. Are some physical concepts more appearance-sensitive than others?
6. Does robustness degrade smoothly with augmentation severity or fail abruptly?
7. Do photometric changes behave differently from object-level texture/material changes?
8. Can a counterfactual calibration procedure produce a better physical-surprise metric?
9. Can the proposed diagnostics predict which JEPA checkpoints genuinely generalize to new visual domains?

---

## 5. Novelty position

### What is **not** novel

The following claims should **not** be presented as the contribution:

- using prediction error as a surprise signal;
- evaluating possible versus impossible physical events;
- testing whether JEPA representations are affected by visual perturbations;
- encouraging appearance invariance in JEPA world models;
- measuring generic rollout consistency under visual perturbations.

These directions already have substantial precedent.

### Closest prior work

#### Garrido et al. — intuitive physics via V-JEPA surprise

Garrido et al. evaluate V-JEPA through violation-of-expectation experiments and define surprise using representation-space future-prediction error. This is the direct foundation for the proposed work.

#### PhyLatent — physical invariance collapse

PhyLatent explicitly identifies **physical invariance collapse**, where appearance-only changes produce large latent shifts even when the simulator state is unchanged. It also introduces a Static Visual Invariance Constraint using brightness and per-channel color perturbations.

This makes a generic paper of the form *"Are JEPA representations invariant to color changes?"* insufficiently novel.

#### Action-Conditioned Predictive Consistency (ACPC)

An et al. introduce ACPC to compare clean and visually perturbed histories after rolling them forward under identical action sequences. Their Invariance Radius and Separation Rate diagnose robustness without confusing invariance with representational collapse.

Again, generic visual-perturbation robustness is therefore not enough.

### Proposed novelty

The stronger niche is:

> **Evaluate whether the *violation-of-expectation surprise signal itself* remains physically selective under controlled appearance interventions.**

The proposed work differs from generic latent invariance in several ways:

1. **Target quantity:** physical surprise / violation-of-expectation signal rather than only representation distance or planning consistency.
2. **Matched causal structure:** possible and impossible videos share the same appearance intervention.
3. **Counterfactual decomposition:** explicitly quantify physical signal, appearance leakage, and their interaction.
4. **Physics-specific benchmark:** evaluate across intuitive-physics concepts such as permanence, solidity, continuity, immutability, gravity, support, inertia, and collision where available.
5. **Temporal analysis:** determine whether appearance creates false surprise peaks or suppresses the true violation peak.
6. **Absolute and relative evaluation:** test both matched-pair discrimination and single-video plausibility separation.
7. **Potential calibration method:** use counterfactual appearance ensembles to obtain an appearance-calibrated surprise score.

**Important:** before claiming a new metric in a paper, a broader systematic literature search should still be performed. The metric names below are working names, not guaranteed-unclaimed terminology.

---

## 6. Formal setup

Assume a scene is generated from two broad groups of latent variables:

\[
X = G(P, A),
\]

where:

- \(P\) denotes physical variables: geometry, position, velocity, object identity through time, collisions, support relations, trajectories, etc.;
- \(A\) denotes appearance variables: hue, saturation, texture, material rendering, background, lighting, camera color response, and other nuisance visual factors;
- \(G\) is the renderer / observation-generation process.

Let

\[
V_i^+ = G(P_i^+, A_i)
\]

be a physically plausible video and

\[
V_i^- = G(P_i^-, A_i)
\]

be its matched physical counterfactual containing a violation.

Now apply an appearance intervention

\[
\operatorname{do}(A_i \leftarrow A_i^{(a)})
\]

without altering the physical trajectory:

\[
T_a(V_i^+) = G(P_i^+, A_i^{(a)})
\]

\[
T_a(V_i^-) = G(P_i^-, A_i^{(a)}).
\]

The key requirement is:

\[
P(T_a(V)) = P(V).
\]

The transformation must therefore be **physically semantics-preserving**, not merely label-preserving in an informal computer-vision sense.

---

# 7. Proposed metrics

No single scalar should be trusted initially. The main evaluation should report a **metric suite** separating physical sensitivity from appearance invariance.

---

## 7.1 Physical Surprise Margin (PSM)

For scene pair \(i\) under appearance \(a\),

\[
M_{i,a}
=
S(T_a(V_i^-))
-
S(T_a(V_i^+)).
\]

Interpretation:

- \(M_{i,a}>0\): correct physical ordering;
- \(M_{i,a}\approx0\): model does not discriminate the violation;
- \(M_{i,a}<0\): model considers the plausible video more surprising.

This is the basic physics signal whose robustness we want to study.

---

## 7.2 Counterfactual Physical Surprise Consistency (CPSC)

Define

\[
\mathrm{CPSC}
=
\frac{1}{N|\mathcal A|}
\sum_{i=1}^{N}
\sum_{a\in\mathcal A}
\mathbf 1[M_{i,a}>0].
\]

This measures how often the correct physical ordering survives appearance interventions.

It can also be reported per transformation:

\[
\mathrm{CPSC}_a
=
\frac{1}{N}
\sum_i
\mathbf 1[M_{i,a}>0].
\]

### Interpretation

A genuinely appearance-robust physical surprise signal should maintain high CPSC even for strong but physically valid interventions.

---

## 7.3 Appearance–Physics Interaction (API)

Correct ordering alone is weak. A model may keep \(M>0\) while its physics margin changes dramatically.

Define a difference-in-differences quantity:

\[
\mathrm{API}_{i,a}
=
\left|
M_{i,a}
-
M_{i,0}
\right|,
\]

where \(a=0\) denotes the original appearance.

A scale-normalized form is

\[
\mathrm{nAPI}_{i,a}
=
\frac{
|M_{i,a}-M_{i,0}|
}{
|M_{i,0}|+\epsilon
}.
\]

Lower values are better.

This directly asks:

> **Did an appearance-only intervention change the strength of the model's physical judgment?**

This may be one of the strongest core diagnostics.

---

## 7.4 Appearance Leakage Ratio (ALR)

We also want to compare nuisance-induced surprise variation against the magnitude of the true physics signal.

For a scene \(i\), define within-physics appearance variation:

\[
L_i^+
=
\operatorname{median}_{a,b}
\left|
S(T_a(V_i^+))-S(T_b(V_i^+))
\right|,
\]

\[
L_i^-
=
\operatorname{median}_{a,b}
\left|
S(T_a(V_i^-))-S(T_b(V_i^-))
\right|.
\]

Then

\[
L_i
=
\frac{L_i^+ + L_i^-}{2}.
\]

Define

\[
\mathrm{ALR}_i
=
\frac{
L_i
}{
\operatorname{median}_{a}|M_{i,a}|+\epsilon
}.
\]

Interpretation:

- \(\mathrm{ALR}\ll1\): appearance variation is small compared with the physics signal;
- \(\mathrm{ALR}\approx1\): nuisance appearance is as influential as the physical violation;
- \(\mathrm{ALR}>1\): appearance changes surprise more than the tested physical violation.

This provides a natural, scale-relative notion of **appearance leakage**.

---

## 7.5 Margin Retention (MR)

For each transformation,

\[
\mathrm{MR}_{i,a}
=
\frac{
M_{i,a}
}{
M_{i,0}+\epsilon
}.
\]

Robust behavior should yield values near \(1\).

Because ratios become unstable when the clean margin is near zero, report MR only for sufficiently informative clean examples or use a robust clipped/median statistic.

---

## 7.6 Transformation-wise AUROC

For single-video evaluation, use surprise to classify plausible versus implausible videos independently.

For each appearance transformation \(a\),

\[
\mathrm{AUROC}_a
=
\mathrm{AUROC}
\left(
S(T_a(V)),
y_{\mathrm{physical}}
\right).
\]

This tests whether a global threshold on surprise still separates possible and impossible videos after appearance changes.

This is considerably harder than matched-pair evaluation and should be a major result.

---

## 7.7 Temporal Violation Localization

Suppose the ground-truth violation occurs around frame \(t_i^\star\).

Using the surprise trajectory \(S_t\), define the predicted peak

\[
\hat t_i = \arg\max_t S_t.
\]

Measure localization error

\[
E_{\mathrm{loc}}
=
|\hat t_i-t_i^\star|.
\]

Under appearance transformation \(a\),

\[
E_{\mathrm{loc}}^{(a)}
=
|\hat t_i^{(a)}-t_i^\star|.
\]

This determines whether appearance interventions:

- create false surprise peaks;
- shift the maximum away from the violation;
- suppress the true violation peak;
- introduce persistent surprise across the entire video.

---

## 7.8 Recommended reporting format

Do **not** immediately collapse everything into one number.

The primary scorecard should include:

| Property | Desired direction |
|---|---:|
| Clean pairwise physics accuracy | high |
| CPSC | high |
| Transformed single-video AUROC | high |
| nAPI | low |
| ALR | low |
| Temporal localization error | low |
| Margin retention | near 1 |

A composite metric can be considered later, but the scientific value is initially in showing *how* the model fails.

---

# 8. Hypotheses

## H1 — Appearance-invariant physical abstraction

> JEPA prediction in latent space abstracts away physically irrelevant appearance information sufficiently well that physical surprise remains stable under appearance-preserving interventions.

Predictions:

- high CPSC;
- low nAPI;
- low ALR;
- stable surprise peak near the physical violation;
- small degradation under moderate color/texture transformations.

---

## H2 — Appearance leakage

> JEPA physical-surprise estimates remain partially entangled with superficial visual statistics.

Predictions:

- surprise shifts substantially after hue, material, texture, or background changes;
- ALR becomes comparable to or larger than 1 for some transformations;
- transformed single-video AUROC drops more than matched-pair accuracy;
- some transformations create large surprise even for fully plausible videos.

This is arguably the most interesting failure mode.

---

## H3 — Common-mode appearance shift

> Appearance changes alter absolute prediction difficulty but preserve the relative physical-surprise margin.

Then:

\[
S(T_a(V^+)) - S(V^+) \neq 0,
\]

\[
S(T_a(V^-)) - S(V^-) \neq 0,
\]

but

\[
M_{a}\approx M_0.
\]

This would mean pairwise violation-of-expectation evaluation is robust while absolute single-video surprise is not.

That distinction would be scientifically useful.

---

## H4 — Appearance–physics interaction

> Certain appearance factors alter the model's sensitivity to specific physical concepts.

Examples:

- low object-background contrast may weaken object permanence;
- texture changes may affect shape/immutability;
- lighting may affect collision/contact cues;
- material appearance may influence learned priors about support, rigidity, or motion.

This predicts a significant interaction:

\[
\text{physical concept}
\times
\text{appearance transformation}.
\]

---

## H5 — Model scale and training objective affect invariance

> Larger or more modern JEPA models preserve physical surprise more reliably under appearance interventions.

Compare, where feasible:

- V-JEPA;
- V-JEPA 2;
- other latent predictive world models;
- pixel-prediction baselines;
- optionally Image World Model / UniJEPA-style representations where the inference setup is compatible.

A particularly interesting result would be that raw clean physics performance and appearance-invariant physics performance are **not strongly correlated**.

---

## H6 — Predictor amplification

> Small appearance differences at the encoder level may be amplified by future prediction.

Measure both:

\[
d(E(V),E(T_a(V)))
\]

and the corresponding change in predicted future representations / surprise.

This connects the project to ACPC while retaining a different target: the integrity of the physical-surprise signal.

---

# 9. Experimental design

## Experiment 0 — Reproduce the established surprise baseline

Before introducing new perturbations:

1. obtain the original V-JEPA intuitive-physics evaluation code;
2. reproduce pairwise possible/impossible evaluation;
3. reproduce average and maximum surprise;
4. verify per-property results;
5. reproduce single-video AUROC where possible.

This is mandatory. Otherwise later degradation cannot be attributed confidently to appearance interventions.

### Output

A table such as:

| Model | Dataset | Avg-surprise pair acc. | Max-surprise pair acc. | Single-video AUROC |
|---|---|---:|---:|---:|
| V-JEPA | IntPhys | ... | ... | ... |
| V-JEPA 2 | IntPhys / IntPhys 2 | ... | ... | ... |

---

## Experiment 1 — Global photometric interventions

Apply transformations consistently to all frames.

### Families

- hue rotation;
- channel permutation;
- saturation;
- brightness;
- contrast;
- gamma;
- color temperature;
- grayscale;
- mild camera-response curves.

### Severity sweep

For each family, use several strengths:

\[
\lambda \in
\{\lambda_1,\lambda_2,\lambda_3,\lambda_4,\lambda_5\}.
\]

Plot:

\[
\mathrm{CPSC}(\lambda),
\quad
\mathrm{ALR}(\lambda),
\quad
\mathrm{nAPI}(\lambda),
\quad
\mathrm{AUROC}(\lambda).
\]

This reveals whether failure is gradual or threshold-like.

---

## Experiment 2 — Object-specific appearance interventions

This is stronger than simple global augmentation.

Using masks, simulator controls, or re-rendering, change only object appearance while leaving geometry and motion fixed.

Possible interventions:

- object hue;
- texture;
- albedo;
- material appearance;
- pattern;
- object/background color relationship.

The strongest implementation would alter rendering parameters **before rendering**, rather than editing finished RGB frames.

That gives a cleaner causal statement:

\[
\operatorname{do}(A \leftarrow A')
\]

while keeping simulator state identical.

---

## Experiment 3 — Background and environment interventions

Change factors unrelated to the tested object dynamics:

- floor texture;
- wall texture;
- skybox;
- background color;
- environment theme;
- non-interacting distractor appearance.

These tests determine whether the surprise signal depends on scene context even when the physical event is unchanged.

### Important control

Do not automatically treat all lighting and shadow changes as nuisance variables. Shadows can provide physical cues about depth, support, contact, and motion.

Therefore divide transformations into:

1. **strict nuisance interventions**, where physics-relevant information is preserved;
2. **cue-altering interventions**, analyzed separately.

---

## Experiment 4 — Crossed physics × appearance factorial design

For every underlying scene, create:

\[
\{\text{possible}, \text{impossible}\}
\times
\{a_0,a_1,\ldots,a_K\}.
\]

This produces a controlled factorial experiment.

For each scene:

| Physics | Appearance | Surprise |
|---|---|---:|
| possible | original | \(S_{+,0}\) |
| impossible | original | \(S_{-,0}\) |
| possible | red-shift | \(S_{+,1}\) |
| impossible | red-shift | \(S_{-,1}\) |
| possible | new texture | \(S_{+,2}\) |
| impossible | new texture | \(S_{-,2}\) |

This is the core dataset structure required for a clean decomposition.

---

## Experiment 5 — Difference-in-differences analysis

For every scene and transformation, compute

\[
\Delta\Delta_{i,a}
=
\left[
S(T_a(V_i^-))-S(T_a(V_i^+))
\right]
-
\left[
S(V_i^-)-S(V_i^+)
\right].
\]

If appearance and physics are cleanly disentangled,

\[
\Delta\Delta_{i,a}\approx0.
\]

This is a direct statistical test of whether appearance changes the model's physical-surprise margin.

---

## Experiment 6 — Temporal surprise profiles

Plot

\[
S_t
\]

for:

- clean plausible;
- clean impossible;
- appearance-modified plausible;
- appearance-modified impossible.

Align videos around the physical violation frame \(t^\star\).

Questions:

1. Does the impossible video retain a sharp peak at \(t^\star\)?
2. Does recoloring create an earlier false peak?
3. Does the appearance shift elevate surprise everywhere?
4. Does the model recover after several frames?
5. Does the predictor amplify the initial perturbation?

A compelling paper figure would show the four aligned curves for several physical concepts.

---

## Experiment 7 — Compositional appearance shifts

Single transformations may be easy.

Test compositions:

\[
T =
T_{\mathrm{texture}}
\circ
T_{\mathrm{hue}}
\circ
T_{\mathrm{background}}.
\]

Create in-distribution-to-out-of-distribution severity levels.

This tests whether invariance composes or degrades rapidly when multiple nuisance variables change simultaneously.

---

## Experiment 8 — Physical-concept breakdown

Analyze separately by physical principle.

Possible categories depending on dataset availability:

- object permanence;
- shape / object immutability;
- spatiotemporal continuity;
- solidity;
- gravity;
- support;
- inertia;
- collision;
- occluded versus visible violations.

For every concept \(c\), report

\[
\mathrm{CPSC}_c,\quad
\mathrm{ALR}_c,\quad
\mathrm{nAPI}_c.
\]

The central scientific question becomes:

> Which learned physical concepts are genuinely appearance invariant, and which depend on visual shortcuts?

---

## Experiment 9 — Encoder vs predictor decomposition

Measure invariance at several stages.

### Encoder difference

\[
D_E
=
d(E(V),E(T_a(V))).
\]

### Predicted-future difference

\[
D_P
=
d(P(E(V)),P(E(T_a(V)))).
\]

### Surprise difference

\[
D_S
=
|S(V)-S(T_a(V))|.
\]

This can distinguish:

1. appearance sensitivity already present in the encoder;
2. predictor amplification;
3. predictor contraction / correction;
4. sensitivity that specifically appears in the surprise comparison.

This experiment creates a bridge to ACPC without duplicating its control/planning focus.

---

## Experiment 10 — Comparison with non-JEPA baselines

Where practical, include:

- random / untrained V-JEPA;
- VideoMAE-style pixel reconstruction;
- DINO-feature future prediction;
- V-JEPA variants;
- V-JEPA 2;
- optionally other latent world models.

The key plot is not simply clean performance.

Plot:

\[
x = \text{clean physics accuracy}
\]

against

\[
y = \text{appearance-invariant physics score}.
\]

If the correlation is weak, that becomes a strong result:

> High benchmark accuracy does not necessarily imply appearance-invariant physical understanding.

---

# 10. Ablations

## Surprise aggregation

Compare:

- average surprise;
- maximum surprise;
- top-\(k\) temporal average;
- violation-window surprise, when ground-truth event timing is known.

The original V-JEPA physics work found different behavior for average versus maximum surprise depending on paired versus single-video evaluation, so this ablation is essential.

---

## Distance function

Compare:

\[
L_1,
\quad
L_2,
\quad
1-\cos(\hat z,z),
\]

and, if justified, normalized or whitened latent distances.

Question:

> Is appearance leakage a property of the representation or partly an artifact of the chosen distance metric?

---

## Context length

Vary the number of past frames \(C\).

Appearance sensitivity may decline when the model receives enough temporal evidence to identify invariant object dynamics.

---

## Prediction horizon

Vary \(M\) / future horizon.

Long-horizon prediction may:

- amplify nuisance appearance mismatch;
- amplify true physical inconsistency;
- reveal stronger abstraction.

---

## Frame rate

Evaluate whether robustness changes when motion is temporally subsampled.

This can reveal whether the model depends more on visual appearance when motion cues become weaker.

---

## Transformation severity

Every transformation should be evaluated as a curve rather than only one arbitrary magnitude.

---

## Layer-wise representation analysis

If model internals permit it, measure appearance sensitivity at multiple encoder layers.

Possible hypothesis:

- early layers: strongly appearance-sensitive;
- middle layers: increasing invariance;
- late layers: physical information concentrated;
- predictor output: may re-amplify or suppress nuisance factors.

---

# 11. Statistical analysis

The dataset structure is paired, so use paired statistics whenever possible.

Recommended analyses:

### Bootstrap confidence intervals

Bootstrap over scene identities, not individual transformed videos, to avoid pseudo-replication.

Report 95% confidence intervals for:

- CPSC;
- ALR;
- nAPI;
- AUROC;
- margin retention.

### Paired permutation tests

For comparing two models under identical scene/appearance interventions.

### Mixed-effects model

A useful analysis is:

\[
S
\sim
\text{Physics}
+
\text{Appearance}
+
\text{Physics}\times\text{Appearance}
+
(1|\text{Scene}).
\]

The critical term is:

\[
\text{Physics}\times\text{Appearance}.
\]

A strong interaction indicates that appearance changes the model's response to the physical violation.

For multiple physical concepts:

\[
S
\sim
P + A + C
+ P\times A
+ P\times C
+ A\times C
+ P\times A\times C
+ (1|\text{Scene}).
\]

This allows a rigorous test of concept-specific appearance leakage.

### Multiple comparisons

If many transformation families and physics concepts are tested, use FDR correction or preregister a small number of primary hypotheses.

---

# 12. Critical controls

## Control 1 — Preserve physical information

A transformation must not accidentally remove information necessary to infer physics.

Examples of dangerous transformations:

- extreme blur;
- destroying object boundaries;
- eliminating contact shadows;
- making an object visually indistinguishable from the background;
- changing frame timing;
- altering geometry;
- introducing motion artifacts.

These should either be excluded from the strict-invariance set or reported separately as **information-destroying/cue-altering interventions**.

---

## Control 2 — Apply identical appearance intervention to the matched pair

For a possible/impossible pair, the same intervention parameters must be used.

Otherwise the experiment confounds physical status and appearance.

---

## Control 3 — Exact temporal consistency

Appearance edits must remain temporally coherent.

Frame-independent random color jitter can create artificial flicker, which is itself a temporal anomaly and therefore a legitimate source of surprise.

Use temporally fixed transformation parameters unless temporal appearance variation is explicitly being studied.

---

## Control 4 — Transformation quality

For object-level recoloring or texture transfer, confirm that:

- masks do not jitter;
- object boundaries are preserved;
- no new artifacts appear;
- compression settings remain constant.

Simulator rerendering is preferable when available.

---

## Control 5 — Collapse check

A model that ignores everything can appear perfectly invariant.

Therefore invariance must **always** be paired with physics discrimination.

This is why CPSC / AUROC and ALR / nAPI should be reported together.

---

# 13. Potential method extension: Appearance-Calibrated Surprise

If the diagnostic reveals substantial appearance leakage, the project can move from evaluation to method development.

Let \(\mathcal T\) be a set of physically preserving appearance interventions.

Define an ensemble surprise:

\[
S_{\mathrm{ens}}(V)
=
\operatorname{median}_{T\in\mathcal T}
S(T(V)).
\]

This may suppress idiosyncratic appearance-sensitive prediction errors.

A more targeted score could use a robust lower/trimmed estimate:

\[
S_{\mathrm{cal}}(V)
=
\operatorname{TrimmedMean}_{T\in\mathcal T}
S(T(V)).
\]

Then compare:

\[
\mathrm{AUROC}(S)
\quad\text{vs}\quad
\mathrm{AUROC}(S_{\mathrm{cal}})
\]

under unseen appearance shifts.

### Stronger training-time extension

If evaluation reveals systematic leakage, train with a loss encouraging **surprise-margin consistency**:

\[
\mathcal L_{\mathrm{SMC}}
=
\left|
M_{i,a}
-
M_{i,0}
\right|.
\]

This is distinct from aligning arbitrary latent representations: it explicitly preserves the **physics decision margin**.

One could combine:

\[
\mathcal L
=
\mathcal L_{\mathrm{JEPA}}
+
\lambda
\mathcal L_{\mathrm{SMC}}.
\]

However, this should be considered a second-stage contribution. The first paper can be strong even if it is primarily a diagnostic/benchmark paper, provided the empirical finding is substantial.

---

# 14. Expected result patterns

## Outcome A — Strong invariance

Clean and transformed performance remain nearly identical:

\[
\mathrm{CPSC}\approx1,\qquad
\mathrm{ALR}\ll1,\qquad
\mathrm{nAPI}\approx0.
\]

Interpretation:

> V-JEPA's physical surprise is genuinely abstract with respect to the tested nuisance appearances.

This is a positive result, though the paper would need enough diversity to make the claim meaningful.

---

## Outcome B — Pairwise robust, absolute surprise fragile

The physical margin remains stable, but absolute surprise shifts strongly:

\[
M_a\approx M_0
\]

while

\[
S(T_a(V)) \gg S(V).
\]

Interpretation:

> JEPA contains a stable relative physics signal, but raw surprise is not an appearance-independent scalar measure of plausibility.

This is a subtle and publishable result because it changes how surprise-based benchmarks should be interpreted.

---

## Outcome C — Physical margin collapses under appearance changes

\[
\mathrm{CPSC}\downarrow,
\qquad
\mathrm{nAPI}\uparrow,
\qquad
\mathrm{ALR}\gtrsim1.
\]

Interpretation:

> Apparent intuitive-physics performance is partly contingent on visual appearance and cannot be interpreted as fully abstract physical understanding.

This is probably the strongest critical result.

---

## Outcome D — Concept-specific robustness

For example:

- permanence remains robust;
- support becomes fragile under texture/material change;
- collision becomes fragile under lighting;
- continuity remains robust under global hue but fails under background randomization.

Interpretation:

> "Physical understanding" is not a unitary capability; different physical concepts are represented with different degrees of appearance abstraction.

This could be an especially strong scientific story.

---

# 15. Minimal viable project

If compute or engineering time is limited, the project can begin with a small but rigorous experiment.

### Phase 1

Use one frozen pretrained V-JEPA model.

### Dataset

Use matched possible/impossible videos from IntPhys or another supported benchmark.

### Transformations

Start with:

1. hue rotation;
2. saturation;
3. brightness;
4. channel permutation;
5. global grayscale.

Use 5 severity levels where meaningful.

### Metrics

Only:

- clean pairwise accuracy;
- CPSC;
- nAPI;
- ALR;
- transformation-wise single-video AUROC;
- surprise curves.

### Success criterion

The project becomes worth expanding if at least one physically valid transformation produces a reproducible, statistically significant distortion in the physical-surprise margin.

If global photometric transformations do almost nothing, move immediately to **object-specific texture/material and background interventions**, which are substantially more informative.

---

# 16. Strong full-scale version

A stronger paper would construct a dedicated counterfactual benchmark.

Each base physical scenario would have:

- physically plausible trajectory;
- matched impossible trajectory;
- multiple object colors;
- multiple object textures/materials;
- multiple backgrounds;
- multiple lighting conditions;
- optional camera-rendering styles.

Because these are rendered from identical simulator states, the benchmark provides direct intervention-level control.

The resulting data tensor is approximately:

\[
\text{scene}
\times
\text{physical condition}
\times
\text{appearance}
\times
\text{severity}.
\]

This would enable unusually clean causal analysis of learned world-model surprise.

---

# 17. Figures that would make the paper convincing

## Figure 1 — Concept

Two physically identical plausible videos with different appearances and two matched impossible variants.

Diagram:

\[
(P^+,A_1)
\quad
(P^-,A_1)
\]

\[
(P^+,A_2)
\quad
(P^-,A_2)
\]

with the desired property

\[
S(P^-,A)-S(P^+,A)
\]

remaining stable across \(A\).

---

## Figure 2 — Surprise trajectories

Four curves aligned around the violation:

- plausible / original;
- impossible / original;
- plausible / transformed;
- impossible / transformed.

---

## Figure 3 — Robustness curves

x-axis: transformation severity  
y-axis: CPSC / AUROC / nAPI.

---

## Figure 4 — Appearance leakage heatmap

Rows: physical concepts.  
Columns: transformation families.  
Cells: ALR or nAPI.

---

## Figure 5 — Clean performance versus invariant performance

Each point is a model/checkpoint.

x-axis:

\[
\text{clean physics accuracy}
\]

y-axis:

\[
\text{appearance-invariant physics score}.
\]

This directly tests whether standard benchmark performance predicts robust physical abstraction.

---

# 18. Main risks

## Risk 1 — Novelty overlap

The August 2026 PhyLatent and ACPC papers make generic JEPA appearance invariance a crowded direction.

### Mitigation

Center the contribution on:

- physical-surprise integrity;
- violation-of-expectation evaluation;
- physics × appearance interaction;
- absolute vs relative surprise;
- counterfactual rendered pairs;
- temporal localization of physical violations.

---

## Risk 2 — Transformations are not truly physics-preserving

For example, changing apparent material can alter a human observer's expectation of mass, friction, elasticity, or rigidity.

### Mitigation

Distinguish:

- **rendering nuisance**: color/albedo/background;
- **semantic material cues**: metal, rubber, glass, cloth.

The second category is scientifically interesting but should not automatically be called physically irrelevant.

---

## Risk 3 — Benchmark shortcut artifacts

If possible and impossible videos differ in rendering artifacts created by the simulator, a model may exploit them.

### Mitigation

Use matched counterfactual rendering and inspect whether transformations preserve/remove such artifacts.

---

## Risk 4 — Surprise scale is not calibrated across videos

Raw prediction difficulty depends on scene complexity.

### Mitigation

Report:

- relative paired margins;
- single-video AUROC;
- within-scene normalized metrics;
- counterfactual difference-in-differences.

---

## Risk 5 — Color constancy itself can be a tested physical property

Some intuitive-physics work treats arbitrary object color change over time as an object-immutability violation.

### Mitigation

Our appearance intervention must be **temporally consistent**. Recolor the object for the entire video. Do not introduce a mid-video color change unless color constancy is intentionally being tested.

---

# 19. Research contribution statement

A strong final contribution statement could be:

> We introduce a counterfactual evaluation framework for determining whether latent-world-model surprise reflects physical violations independently of visual appearance. Unlike prior violation-of-expectation evaluation, which asks whether impossible events produce larger prediction error, our protocol intervenes on nuisance appearance while preserving the underlying physical trajectories and measures whether the physical-surprise margin itself remains stable. We propose complementary diagnostics for counterfactual physical-surprise consistency, appearance leakage, physics–appearance interaction, and temporal violation localization, and use them to characterize when JEPA-style world models exhibit robust physical abstraction versus appearance-dependent surprise.

---

# 20. Concrete research roadmap

## Stage 1 — Baseline
- reproduce V-JEPA surprise evaluation;
- verify clean IntPhys / available benchmark results;
- implement frame-level surprise extraction.

## Stage 2 — Simple interventions
- implement deterministic video-consistent photometric transformations;
- run severity sweeps;
- calculate CPSC, nAPI, ALR, AUROC.

## Stage 3 — Diagnose failures
- per-physics-concept analysis;
- temporal surprise curves;
- encoder versus predictor sensitivity;
- average versus max surprise.

## Stage 4 — Strong counterfactual data
- obtain simulator source or construct a small controlled simulator;
- rerender identical trajectories under object/background appearance changes;
- verify exact state equality.

## Stage 5 — Compare models
- V-JEPA variants;
- V-JEPA 2 where evaluation is technically compatible;
- pixel-prediction baseline;
- additional latent predictive models.

## Stage 6 — Optional method
- appearance-ensemble calibrated surprise;
- surprise-margin consistency loss;
- evaluate whether calibration/training improves robustness on held-out transformations.

---

# 21. What would make this paper genuinely strong?

The paper becomes much stronger if it can demonstrate one of the following:

1. **Large hidden failure:** clean physics accuracy is high but collapses under physically irrelevant appearance changes.
2. **Metric mismatch:** pairwise relative surprise is robust while absolute single-video surprise is heavily appearance-dependent.
3. **Concept-specific structure:** some physical concepts are abstract while others rely on appearance cues.
4. **Model-ranking reversal:** the model with the best standard physics benchmark score is not the model with the best appearance-invariant physics score.
5. **Predictive diagnostic:** the proposed metric predicts out-of-domain physics performance better than ordinary clean accuracy.
6. **Simple correction:** counterfactual calibration significantly improves physics discrimination under unseen appearance shifts.

The weakest possible result would be merely:

> "V-JEPA changes its embeddings when we recolor images."

That is already too close to existing robustness/invariance work.

The target result should instead answer:

> **Does the model's *belief that an event violates physics* survive interventions that change how the world looks without changing what physically happens?**

---

# 22. References

## Core JEPA papers

**[1] Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., & Ballas, N. (2023).**  
*Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.*  
ICCV 2023.  
https://arxiv.org/abs/2301.08243

**[2] Bardes, A., Garrido, Q., Ponce, J., Chen, X., Rabbat, M., LeCun, Y., Assran, M., & Ballas, N. (2024).**  
*Revisiting Feature Prediction for Learning Visual Representations from Video.*  
Introduces V-JEPA.  
https://arxiv.org/abs/2404.08471

**[3] Assran, M., Bardes, A., Fan, D., Garrido, Q., Howes, R., et al. (2025).**  
*V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning.*  
https://arxiv.org/abs/2506.09985

---

## Physical surprise and intuitive physics

**[4] Garrido, Q., Ballas, N., Assran, M., Bardes, A., Najman, L., Rabbat, M., Dupoux, E., & LeCun, Y. (2025; revised 2026).**  
*Intuitive physics understanding emerges from self-supervised pretraining on natural videos.*  
Defines a V-JEPA surprise signal based on latent future-prediction error and evaluates violation-of-expectation physics.  
https://arxiv.org/abs/2502.11831

**[5] Riochet, R., Castro, M. Y., Bernard, M., Lerer, A., Fergus, R., Izard, V., & Dupoux, E. (2018).**  
*IntPhys: A Framework and Benchmark for Visual Intuitive Physics Reasoning.*  
https://arxiv.org/abs/1803.07616

**[6] Bordes, F., Garrido, Q., Kao, J. T., Williams, A., Rabbat, M., & Dupoux, E. (2025; revised 2026).**  
*IntPhys 2: Benchmarking Intuitive Physics Understanding in Complex Synthetic Environments.*  
https://arxiv.org/abs/2506.09849

---

## Appearance, invariance, and JEPA world models

**[7] Garrido, Q., Assran, M., Ballas, N., Bardes, A., Najman, L., & LeCun, Y. (2024).**  
*Learning and Leveraging World Models in Visual Representation Learning.*  
Introduces Image World Models and studies prediction of global photometric transformations in latent space.  
https://arxiv.org/abs/2403.00504

**[8] Zeng, X., Ren, H., & Song, Z. (2026).**  
*PhyLatent: Learning Dynamics-Relevant Representations for JEPA World Models.*  
Introduces physical invariance collapse and a training-time static visual invariance constraint.  
https://arxiv.org/abs/2608.05720

**[9] An, G., Wu, Z., Dong, H., Yan, Y., Gui, Z., Chen, H., Ruan, S., Wang, X., Ling, Y., & Tian, Q. (2026).**  
*Diagnosing JEPA World Models with Action-Conditioned Predictive Consistency.*  
Introduces ACPC, Invariance Radius, and Separation Rate for visual-perturbation robustness in action-conditioned world models.  
https://arxiv.org/abs/2608.12939

**[10] Lanji, A., Liu, D., Li, J., Xu, H., Chen, M., & Tian, Y. (2026).**  
*UniJEPA: A Unified Joint-Embedding Predictive Architecture for Task-Agnostic Visual World Modeling.*  
Combines photometric and temporal prediction in a shared JEPA framework.  
https://arxiv.org/abs/2608.07409

---

# 23. Short abstract draft

We investigate whether the physical expectations encoded by joint-embedding predictive world models are invariant to changes in visual appearance. Prior work has shown that V-JEPA exhibits violation-of-expectation behavior: physically implausible videos produce larger latent future-prediction errors than matched plausible videos. However, this surprise signal may conflate violations of physical dynamics with changes in nuisance visual variables. We propose a counterfactual evaluation framework in which the underlying physical trajectories are held fixed while object color, texture, background, illumination, and related appearance variables are systematically intervened upon. We evaluate whether the model preserves its physical-surprise margin across these counterfactual views and introduce complementary diagnostics for Counterfactual Physical Surprise Consistency, Appearance–Physics Interaction, Appearance Leakage Ratio, and temporal violation localization. This framework distinguishes models that are simply predictive on familiar visual distributions from models whose violation-of-expectation signal reflects a more appearance-invariant representation of physical dynamics.

---

# 24. Bottom line

The research direction should **not** be framed as:

> "Can JEPA models handle changed colors?"

It should be framed as:

> **"Does the physical-surprise signal of a predictive world model remain invariant under causal interventions on appearance?"**

The central object of study is therefore not generic augmentation robustness, but the **causal validity of surprise as evidence for learned physical understanding**.
