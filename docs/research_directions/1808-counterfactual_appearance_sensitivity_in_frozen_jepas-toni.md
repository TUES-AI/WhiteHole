# Counterfactual Appearance Sensitivity in Frozen JEPA Representations

**Author:** Toni  
**Date:** 18 August 2026  
**Status:** Provisional research direction for discussion; no experimental direction has been selected.

## Core question

> When the physical state and trajectory are held exactly fixed while visual appearance changes, what changes inside a frozen visual world model, and when do those changes stop affecting physics/content-relevant computation?

Interesting: │ JEPA may suppress unpredictable pixel detail while preferentially retaining stable appearance.

The project should distinguish three properties that are often conflated:

1. **Appearance accessibility:** can appearance still be decoded from the representation?
2. **Physical-state accessibility:** can the same state or trajectory still be recovered across appearances?
3. **Functional dependence:** does appearance alter native prediction, a frozen downstream readout, action ranking, or planning?

Appearance need not be deleted for a representation to be useful. It may remain decodable while becoming functionally decoupled from physical computation.

## Experimental unit

For physical state or trajectory \(S\) and appearance \(A\):

```text
anchor:   (S,  A)
positive: (S,  A')   same physics, different appearance
negative: (S', A)    different physics, same appearance
```

Physics must be generated once and rerendered without changing geometry, pose, velocity, contacts, camera, timing, mass, friction, restitution, or actions.

Core appearance interventions:

- object albedo and texture;
- visual material while physical material parameters remain fixed;
- illumination color/intensity;
- background/environment appearance;
- global camera response such as white balance or gamma.

Sensor corruption, observation geometry, and temporal intervention should be separate benchmark families rather than grouped under appearance.

## Main measurements across depth

At every encoder layer, for pooled and patchwise representations:

1. appearance-factor probes;
2. physical-state/content probes;
3. source-trained probes evaluated unchanged across appearances;
4. appearance-specific oracle probes;
5. same-state cross-appearance retrieval with hard physical negatives;
6. normalized appearance displacement versus physical-state displacement;
7. native JEPA predictor sensitivity where a predictor is released.

Where action-conditioned world models are available, add action-ranking agreement and planning success. Representation similarity alone is not a functional result.

The source-versus-oracle distinction diagnoses different failures:

- source probe fails but oracle succeeds: state information remains but moved coordinates;
- both fail: accessible state information was lost;
- appearance remains decodable while native behavior stays stable: functional decoupling;
- appearance remains decodable and behavior changes: functional entanglement.

## Models

Image track:

- I-JEPA;
- MAE;
- DINOv2 or DINOv3.

Video track:

- V-JEPA 2/2.1;
- VideoMAE-v2;
- optionally VideoPrism or a DINO frame-feature baseline.

Action-conditioned validation:

- LeWorldModel or DINO-WM/PLDM-style planning environments.

Public-checkpoint comparisons describe model families. They cannot establish that latent prediction caused a difference because training data, scale, architecture, augmentation, and compute remain confounded.

## Data

A practical progression is:

1. **CRONOS** for an existing controlled video pilot involving collisions, falls, occlusion, appearance, scene, object, and viewpoint variants;
2. **PUG-ImageNet** for controlled image texture, environment, and lighting;
3. a dedicated repeated-render simulator set if CRONOS does not provide sufficient appearance diversity or exact state matching for the required factors;
4. LeWorldModel environments for functional prediction/planning validation.

## Closest novelty constraints

- **SimMotion** already tests same-motion retrieval under visual changes.
- **Latent Video Prediction Learns Better World Models** already benchmarks V-JEPA robustness to corruption, occlusion, contact cues, and temporal order.
- **Interpreting Physics in Video World Models** already performs layerwise physics probing and interventions.
- **PhyLatent** already defines physical invariance collapse by comparing appearance displacement with real state displacement in a final LeWorldModel latent.
- **CRONOS** already evaluates counterfactual physical consistency of video generators under controlled visual interventions.
- **Substance or Style** already probes non-semantic information in frozen image embeddings.

The remaining intersection is a full-depth, exact-pair audit of frozen image/video JEPA foundation models that separates information accessibility from native functional dependence.

## Possible contribution

A credible contribution would be:

> A factor-controlled benchmark and mechanistic audit showing where appearance remains accessible across frozen JEPA depth, whether physical state remains usable across exact appearance counterfactuals, and whether retained appearance affects native predictor or downstream computation.

The result may be positive or negative:

- a depth-localized loss of appearance accessibility;
- appearance retained but functionally decoupled;
- content preserved in appearance-specific coordinates;
- no meaningful transition, challenging the usual abstraction intuition;
- similar behavior across JEPA and reconstruction models, weakening objective-specific claims.

## Relationship to Alex’s proposal

Alex’s appearance-invariant physical-surprise direction is a compatible, narrower functional track. It asks whether the plausible-versus-impossible prediction-error margin survives appearance changes. In this direction, physical surprise would be one important native readout alongside state accessibility, retrieval, and planning rather than the sole object of study.

## Boundaries

- Do not assume an “Appearance-Discard Zone” before observing one.
- Do not use generic corruption robustness as the central contribution.
- Do not treat cosine/CKA stability as evidence of usable invariance.
- Do not equate probe decodability with causal use.
- Do not start with adapter development or model retraining.
- Do not update the main project direction until the team selects among the competing proposals.
