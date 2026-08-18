# Appearance-Discard Zones in JEPA Representations

## Core Idea

The goal of this project is to test whether **I-JEPA, V-JEPA, and V-JEPA 2 actually learn to separate appearance from meaningful structure**, as their latent-prediction objective is often intuitively expected to do.

More specifically, we ask:

> **When, where, and to what extent do JEPA representations become invariant to appearance while preserving semantic and physically relevant information?**

Here, **appearance** includes properties such as:

- texture,
- color,
- lighting,
- artistic style,
- background appearance,

while **content / structure** includes:

- object identity,
- shape,
- action,
- motion,
- trajectory,
- physically relevant scene structure.

The project uses only **frozen, publicly released checkpoints**. No JEPA pretraining or fine-tuning is required.

---

## Main Motivation

JEPA models predict in **latent space instead of pixel space**.

The motivation behind latent prediction is that the model does not need to reproduce every low-level visual detail. Instead, it can focus on information that is useful and predictable, such as:

- object identity,
- motion,
- geometry,
- action,
- physical structure.

However, it is not clear whether JEPA models actually discard appearance information internally.

A model may still contain information about texture, color, or style even if these properties no longer affect its downstream computation.

This creates an important distinction:

### Information being present

Appearance can still be decoded from an internal representation.

### Information being functionally used

Appearance can still causally affect later model computation.

The project therefore studies both.

---

# Central Hypothesis

There may exist a region in JEPA networks where appearance information becomes less important while semantic and physical information remains strong.

We call this the:

# **Appearance-Discard Zone**

This is inspired by the **Physics Emergence Zone** reported in recent V-JEPA 2 interpretability work.

That work asks:

> At what depth does physical information emerge?

Our project asks the complementary question:

> At what depth does nuisance appearance information stop mattering?

The result does not have to be literal information deletion.

A potentially more interesting outcome is:

> Appearance remains linearly decodable, but becomes causally irrelevant to downstream representations.

If that happens, the phenomenon may be better described as an:

- **Appearance Suppression Zone**, or
- **Appearance Decoupling Zone**.

---

# Main Experimental Design

The paper has two central pillars.

---

## Pillar 1: Layerwise Probing

We extract representations from every transformer layer of frozen JEPA models.

For the same underlying image or video, we create appearance-modified versions while keeping the semantic content unchanged.

For example:

```text
Original:
brown dog running

Appearance-modified:
purple stylized dog running
```

The dog identity and motion remain the same, while its appearance changes.

Possible appearance transformations include:

- color / hue shifts,
- texture changes,
- style transfer,
- lighting / contrast changes,
- grayscale,
- background appearance changes,
- material or surface changes.

For every layer \(l\), we obtain:

\[
z_l = f_l(x)
\]

for the original sample and

\[
z'_l = f_l(T_a(x))
\]

for the appearance-modified sample.

We then train two families of cheap linear probes.

### Appearance probe

Predicts which appearance transformation was applied:

\[
P_l^{appearance}(z_l) \rightarrow a
\]

Examples:

- original vs stylized,
- texture A vs texture B,
- color shift category,
- lighting condition.

### Content probe

Predicts semantic or physically meaningful information:

\[
P_l^{content}(z_l) \rightarrow y
\]

Examples:

- object identity,
- object category,
- action class,
- motion type,
- trajectory,
- scene content.

We track both probe accuracies across network depth.

The ideal pattern would look roughly like:

```text
Early layers:
appearance high
content moderate

Middle layers:
appearance starts decreasing
content rises

Late layers:
appearance reduced
content remains high
```

The transition region would be the candidate **Appearance-Discard Zone**.

---

# Pillar 2: Causal Activation Intervention

Linear probes alone are not enough.

A probe can decode information that the network itself does not actually use.

Therefore, we test whether appearance information is **causally important**.

Rather than replacing the entire hidden state between clean and appearance-modified inputs, we identify an **appearance-related subspace** at each layer.

Let:

\[
A_l
\]

be the appearance subspace at layer \(l\).

We can estimate it from appearance probe directions or related subspace methods.

Then we intervene only on that component.

For example:

\[
h_l' =
h_l(x_{styled})
-
P_{A_l}h_l(x_{styled})
+
P_{A_l}h_l(x_{clean})
\]

where \(P_{A_l}\) projects onto the appearance subspace.

This replaces only the appearance-related component while preserving the rest of the representation as much as possible.

We then measure how much later representations or predictions change.

---

# Why the Causal Experiment Matters

This experiment allows us to distinguish between several possible situations.

## Case 1: Appearance is decodable and causal

The model still actively uses appearance.

```text
Appearance information
        ↓
representation
        ↓
downstream computation
```

## Case 2: Appearance is no longer decodable

The model genuinely removes much of the appearance information.

```text
Appearance
   ↓
mostly erased
```

## Case 3: Appearance remains decodable but is no longer causal

This may be the most interesting outcome.

```text
Appearance
    ↓
representation
    ↓
still recoverable by probe

but

appearance component
    X
downstream computation
```

This would suggest that JEPA does not need to literally erase appearance.

Instead, the network may **quarantine or decouple appearance information from the semantic and physical computation used downstream**.

---

# Latent-vs-Pixel Comparison

A very important part of the project is comparing JEPA models against pixel-reconstruction models.

Otherwise, we can only conclude:

> This is how JEPA behaves.

We cannot strongly connect the result to the difference between **latent prediction and pixel prediction**.

The clean comparison is:

## Image models

- **I-JEPA** — latent prediction
- **MAE** — pixel reconstruction

## Video models

- **V-JEPA 2** — latent prediction
- **VideoMAE / VideoMAE-v2** — pixel reconstruction

Optional additional control:

- **DINOv2** — discriminative self-supervised representation learning

The key question becomes:

> Does latent prediction cause appearance information to become irrelevant earlier or more strongly than pixel reconstruction?

For example, if MAE continues to strongly encode texture in late layers while I-JEPA suppresses its causal influence, that would directly support the idea that latent prediction changes what visual information the representation prioritizes.

---

# Supporting Experiment: Texture-vs-Shape Bias

A supporting behavioral experiment can use a Stylized-ImageNet / cue-conflict-style protocol.

The goal is to test whether JEPA models behave more according to:

- **texture**, or
- **shape**.

Example:

```text
Image shape: cat
Texture: elephant
```

A texture-biased model may predict:

```text
elephant
```

A shape-biased model may predict:

```text
cat
```

This places JEPA models on the established texture-vs-shape spectrum.

However, this is a **supporting experiment**, not the main contribution.

The main contribution is the mechanistic combination of:

1. layerwise information analysis,
2. causal interventions,
3. latent-vs-pixel comparison.

---

# Main Research Question

The strongest overall formulation is:

> **Does latent-space prediction cause visual representations to separate nuisance appearance from semantic and physically relevant structure, and where in the network does this separation occur?**

---

# Possible Main Findings

Several outcomes would all be scientifically interesting.

## Outcome A: Appearance disappears

Appearance probe performance falls sharply while content remains strong.

This would provide direct evidence for an **Appearance-Discard Zone**.

---

## Outcome B: Appearance remains decodable but becomes causally irrelevant

Appearance probes still work, but interventions on the appearance subspace stop affecting later computation.

This would suggest an **Appearance-Decoupling Zone**.

This may actually be the strongest and most mechanistically interesting result.

---

## Outcome C: Appearance remains both decodable and causal

Then JEPA does not strongly separate appearance from content in the way commonly assumed.

This would challenge a common intuition about latent prediction.

---

## Outcome D: JEPA and pixel models behave similarly

Then latent prediction may not itself explain appearance invariance.

This would also be an informative negative result.

---

# Models

Primary models:

- I-JEPA
- V-JEPA 2

Optional:

- original V-JEPA

Controls:

- MAE
- VideoMAE / VideoMAE-v2

Optional additional comparison:

- DINOv2

All models remain completely frozen.

---

# Compute Requirements

The project is intentionally designed to be cheap.

No model pretraining is required.

No model fine-tuning is required.

The main computation consists of:

- forward passes through frozen models,
- storing intermediate activations,
- training linear probes,
- running activation interventions.

This makes the project realistic within roughly two weeks.

---

# Minimal Experiment Set

A strong workshop paper likely only needs:

1. I-JEPA
2. V-JEPA 2
3. MAE
4. VideoMAE-v2
5. 4–5 controlled appearance transformations
6. layerwise appearance probes
7. layerwise content/action probes
8. appearance-subspace causal interventions
9. texture-vs-shape evaluation as supporting evidence

The project should avoid adding unnecessary experiments unless the main results are already complete.

---

# Main Figures

## Figure 1: Layerwise Appearance vs Content

Plot across model depth:

```text
Probe accuracy
100% ┤
     │ Appearance ───────╲
 80% │                   ╲
     │                    ╲
 60% │                     ╲____
     │
 40% │          Content ─────────────
     │       ╱
 20% │_____╱
     └───────────────────────────────
       early      middle       late
```

The crossing or transition region becomes the candidate Appearance-Discard Zone.

---

## Figure 2: Causal Influence Across Layers

Plot the effect of appearance-subspace interventions across depth.

For example:

```text
Causal effect
High ┤\
     │ \
     │  \
     │   \
Low  │    \________________
     └──────────────────────
       early  middle  late
```

If causal influence collapses while appearance remains decodable, that provides evidence for representational decoupling.

---

## Figure 3: JEPA vs Pixel Reconstruction

Compare models directly:

```text
Appearance influence

MAE / VideoMAE
high ─────────────────────────

JEPA
high ───────╲
             ╲
              ╲________ low
```

This would directly connect the results to the **pixels-vs-latents** question.

---

# Novelty

The central novelty is not merely testing robustness.

The project studies the **mechanism** by which JEPA representations may become invariant.

The proposed contribution is:

> A layerwise and causal characterization of how appearance information evolves inside JEPA models, including whether it remains decodable, whether it remains functionally used, and whether latent-predictive models separate appearance from semantic and physical structure differently from pixel-reconstruction models.

This is closely inspired by recent physics interpretability work on V-JEPA 2, but applies the methodology to the complementary question of **appearance suppression rather than physics emergence**.

---

# Workshop Framing

The project is especially well suited for the **NeurIPS 2026 World Models in Physical AI workshop**.

It directly addresses the question:

> **Pixels vs. latents: what does latent prediction actually buy us?**

Instead of evaluating only reconstruction quality or downstream accuracy, the project asks whether latent prediction causes a meaningful internal separation between:

```text
nuisance visual appearance
            vs.
semantic / physically relevant structure
```

The causal intervention component also makes the benchmark more mechanistic than a standard robustness evaluation.

---

# Intended Contribution

The final paper would argue one of two likely conclusions.

### Strong invariance result

> JEPA models contain a depth-localized Appearance-Discard Zone in which appearance information rapidly decreases while semantic and physical information is preserved.

or, potentially more interestingly:

### Causal decoupling result

> JEPA models do not fully erase appearance. Instead, they undergo a depth-dependent transition in which appearance remains partially decodable but becomes increasingly causally disconnected from downstream semantic and physical computation.

The second result would show that **invariance does not necessarily require information destruction**.

The model can retain nuisance information while learning not to rely on it.

---

# Target Deliverable

An approximately **6–8 page NeurIPS 2026 workshop paper** for **World Models in Physical AI**, with a possible shorter 4-page version if necessary.

The paper would center around:

- one clear mechanistic question,
- frozen public checkpoints,
- layerwise probing,
- causal activation intervention,
- JEPA-vs-pixel controls,
- a small supporting shape-vs-texture benchmark.

The broader goal is to provide concrete evidence about **what latent-space prediction actually changes inside a visual world model**.
