# Interpreting Physics in Video World Models

**Paper:** Sonia Joseph et al., *Interpreting Physics in Video World Models*, arXiv:2602.07050. [arXiv](https://arxiv.org/abs/2602.07050) · [HTML](https://arxiv.org/html/2602.07050)

**Method:** This summary was prepared from the full arXiv TeX source (main text, appendix, tables, bibliography, and figure captions), fetched on 2026-08-15. Local source: `/tmp/arxiv-src/2602.07050/`.

## Problem and core idea

The paper asks where physical variables become accessible inside frozen video encoders and what representational form they take. It studies V-JEPA 2-L/H/G and VideoMAE-v2, rather than only evaluating final-layer behavior.

Its central result is a sharp transition near one-third of encoder depth, called the **Physics Emergence Zone**. Possible-versus-impossible IntPhys classification and motion direction become linearly accessible there, peak at intermediate depth, and weaken toward the output. Scalar speed and acceleration magnitude are accessible earlier.

The authors argue against a compact, reusable “physics engine” representation. Motion direction and IntPhys discrimination co-emerge but occupy nearly random-overlap subspaces; direction is encoded as a distributed, high-dimensional circular population code rather than a small set of state variables.

## Method details

- **IntPhys:** matched possible/impossible videos differing at a breakpoint; probes distinguish object-permanence, shape-constancy, and continuity violations.
- **Controlled Kubric motion:** 392 constant-velocity videos and 280 constant-acceleration videos, each 16 frames at 24 fps. Factors include eight directions, several speed/acceleration magnitudes, and seven start positions.
- **Layerwise probing:** linear probes on mean-pooled space-time tokens, complemented by attentive-MLP probes. The main probe sweep uses 20 learning-rate/weight-decay settings and five-fold grouped cross-validation.
- **Patch analysis:** per-patch and cross-spatial-region probes test whether direction changes from local/retinotopic to globally available.
- **Subspace analysis:** probe-weight bases are compared with principal angles, projection overlap, Grassmann distance, and dimension-matched random baselines.
- **Circuit intervention:** nearby spatial and/or temporal attention weights are zeroed and renormalized within the emergence zone.
- **Direction steering:** iterative orthogonal probes estimate a direction subspace; activations are replaced within that subspace and read by a held-out evaluation probe.

## Key results

- V-JEPA 2-L/H/G rises from approximately chance to roughly 85–95% IntPhys probe accuracy near one-third depth. VideoMAE-v2-G shows a related transition, while smaller VideoMAE-v2 variants do not reliably do so.
- Intermediate layers preserve more IntPhys information than final layers. A newly trained V-JEPA predictor also performs best from middle-layer features, although even early features can become useful once the predictor learns on top of them.
- Speed and acceleration magnitude are decodable early; direction becomes reliable around the emergence zone. Direction similarly emerges in multi-object CLEVRER scenes.
- Direction and IntPhys subspaces have mean principal angles around 69–75 degrees, and their 7–13% projection overlap matches dimension-controlled random baselines. Speed/IntPhys overlap is below 3% and also near random.
- At the transition, direction becomes decodable from individual patches and generalizes across spatial regions, consistent with a local-to-global representational change.
- Suppressing local attention in the emergence zone strongly harms direction and IntPhys readouts while affecting ImageNet classification much less. Combined local spatial/temporal suppression reduces direction R² from 0.97 to 0.14; temporal suppression reduces IntPhys from 78.3% to 51.9%.
- Iterative erasure suggests direction occupies tens of dimensions. Steering with one to five probe directions is weak, while approximately 20 probes move a separately trained held-out probe toward a target direction with about 11.9-degree error.

## What is relevant for WhiteHole

The paper supplies a strong template for **layerwise accessibility, geometry, patch distribution, and coarse functional intervention**, but it does not study controlled appearance counterfactuals. Its methodology can therefore inspire WhiteHole only if the new work contributes more than replacing “physics” labels with “style” labels.

Important constraints on that extension:

1. The Physics Emergence Zone is not uniquely JEPA-specific: VideoMAE-v2-G also exhibits it. A similarly located appearance transition would not by itself establish an effect of latent prediction.
2. Probe decodability is not functional use. The paper addresses this partly through attention ablation and held-out-probe steering, but steering changes what another probe reads rather than demonstrating changed physical prediction, planning, or control behavior. The authors accurately describe their evidence as coarse causal influence rather than a complete circuit mechanism.
3. A visual-shift study should make the causal intervention at the **data-generating process** first: re-render identical scene states and trajectories under independently randomized appearance factors. This gives a stronger appearance-only counterfactual than post-hoc stylization.
4. Frozen public checkpoints can establish model-family behavior, but differences in architecture, pretraining data, scale, and preprocessing prevent a causal claim that the JEPA objective produced the difference.
5. Intermediate-layer peaks can also reflect downstream/pretraining distribution shift, so an “appearance-discard zone” needs controls against generic OOD layerwise degradation.

## Concrete experiments to run next

- Re-render identical Kubric trajectories under factorial changes to object albedo/texture, illumination, and background while locking meshes, camera, poses, contacts, friction, and temporal sampling.
- At every layer, jointly measure appearance-factor decoding and physics-label decoding with all renderings of one trajectory assigned to the same split.
- Train a clean-domain physics probe and a per-appearance oracle probe. Clean-to-shift degradation with preserved oracle accuracy identifies coordinate drift; degradation of both indicates information loss.
- Repeat on I-JEPA/MAE using controlled PUG-ImageNet factors, with camera/object orientation treated as geometry controls rather than appearance.
- If using activation interventions, report completeness of appearance removal, selectivity for physics/content, matched-rank random-subspace controls, and an independent downstream behavior—not only the probe used to define the subspace.

## Risks / open questions

- “Appearance” is task-dependent: material, color, shadows, and lighting can carry physical or identity information. The benchmark must declare which target variables make each factor a nuisance.
- Style transfer can alter silhouettes, local geometry, object identity, and temporal coherence; it is unsuitable as the primary appearance-only intervention.
- Mean pooling can hide patch-local information, while attentive probes can learn the task themselves. Both should be reported under a fixed, nested validation protocol.
- The paper does not release a repository link in its TeX source, so exact reproduction details beyond the manuscript are currently unavailable.
