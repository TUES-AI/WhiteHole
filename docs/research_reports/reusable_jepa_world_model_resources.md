# Reusable JEPA and world-model resource inventory

**Recorded:** 2026-08-15  
**Purpose:** Operational inventory for the proposed frozen-world-model appearance/physics analysis. This is not a decision to adopt that direction and is not experimental evidence.

Status terms:

- **Local + artifact:** code and a relevant checkpoint/data artifact were verified locally.
- **Local code:** repository exists locally, but a relevant checkpoint was not found.
- **External:** public resource is not currently mirrored locally.
- Numbers described as **author-reported** have not been reproduced by WhiteHole.

## Highest-value local assets

### bounce2D physics and surprise probes

- **Upstream:** [Mantra20/eb_jepa](https://github.com/Mantra20/eb_jepa), a fork of [facebookresearch/eb_jepa](https://github.com/facebookresearch/eb_jepa)
- **Local repository:** `/Volumes/SSD/repos/worldmodels/Mantra20__eb_jepa`
- **Verified local commit:** `319fa4ff1c920d6ebbd39acb3cfd3686fddab9fb`
- **Status:** Local + artifact
- **Checkpoint:** `runs/main_simt12_seed1/e-300.pth.tar`; the repository warns that the corresponding `latest.pth.tar` is corrupted
- **Useful code:** `bounce2d/`, `probes.py`, `make_data.py`, `scripts/run_probes.py`, `scripts/triviality_test.py`
- **Provides:** bit-exact trajectories; position, velocity, and energy labels; paired teleport, phantom-bounce, energy-gain, and energy-loss violations; trajectory-disjoint probes; latent-prediction surprise.
- **Best reuse:** validate paired same-physics/different-appearance versus same-appearance/different-physics metrics before scaling to foundation models.
- **Limitation:** scratch-trained EB-JEPA on 65×65 two-channel observations, not a pretrained natural-video foundation model. Repository results are not WhiteHole reproductions.

### LeWorldModel predictor and planning protocol

- **Paper:** [LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels](https://arxiv.org/abs/2603.19312)
- **Project:** [le-wm.github.io](https://le-wm.github.io/)
- **Upstream:** [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm)
- **Models and data:** [Hugging Face LeWM collection](https://huggingface.co/collections/quentinll/lewm), including [TwoRooms](https://huggingface.co/quentinll/lewm-tworooms), [Reacher](https://huggingface.co/quentinll/lewm-reacher), [Push-T](https://huggingface.co/quentinll/lewm-pusht), and [Cube](https://huggingface.co/quentinll/lewm-cube)
- **Local repository:** `/Volumes/SSD/repos/le-wm`
- **Verified local commit:** `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`
- **Cached locally:** TwoRooms model and dataset under `/Volumes/SSD/huggingface/huggingface/hub/`
- **Status:** Local + TwoRooms artifacts; other environment artifacts were not found locally
- **WhiteHole summary:** [`../paper_summaries/summary_2603.19312_leworldmodel/summary.md`](../paper_summaries/summary_2603.19312_leworldmodel/summary.md)
- **Best reuse:** native next-latent surprise, rollout divergence, CEM behavior, and planning under paired appearance versus physical interventions.
- **Existing precedent:** the paper compares color changes with teleportation. Its reported color effect is weak while teleportation produces stronger surprise, but sample counts and effect sizes are incomplete; treat this as a protocol seed, not established WhiteHole evidence.

### WhiteHole frozen-adapter infrastructure

- **Repository commit:** `a91162d2724f114bd259bc5c18dafc7ec52a9dd4`
- **Code:** `scripts/jepa_visual_adapters.py`, `scripts/train_ijepa_visual_adapters.py`, `scripts/train_vjepa_visual_adapters.py`
- **Tests:** `tests/test_jepa_visual_adapters.py`, `tests/test_vjepa_visual_adapters.py`
- **Results:** `results/11-17-08-jepa-visual-adapters/`
- **Large checkpoints:** `/Volumes/SSD/whitehole/artifacts/11-jepa-visual-adapters/`
- **Best reuse:** deterministic paired shifts, disjoint data partitions, frozen-core checks, layer extraction patterns, faithful V-JEPA 2.1 student/predictor/teacher loading, and functional-versus-pixel metric separation.
- **Limitation:** the experiment studied adapter repair, not the proposed layerwise counterfactual benchmark. Its Kinetics-mini probe was too small to serve as benchmark evidence.

## Hackathon repositories and environments

### Matched Moving-MNIST intuitive-physics stimuli

- **Upstream:** [HackTheWorlds/result-modelusifyoucan](https://github.com/HackTheWorlds/result-modelusifyoucan)
- **Local repository:** `/Volumes/SSD/repos/worldmodels/HackTheWorlds__result-modelusifyoucan`
- **Verified local commit:** `48fef28b2cf102f169d4f206c050a9285160fe82`
- **Relevant path:** `examples/intuitive_physics/`
- **Status:** Local code; the referenced trained checkpoints point to an unavailable cluster filesystem
- **Provides:** matched plausible/impossible clips, known violation frames, teleport/reversal/pass-through stimuli, per-clip JEPA latent energy, AUROC, SimVP and ConvLSTM controls, plotting scripts.
- **Best reuse:** add exact appearance-only controls such as digit palette, texture, background, and identity changes while preserving trajectories; retain physically legal fast-motion controls to separate physics sensitivity from generic novelty.
- **Evidence warning:** checked-in figures and reports are author-produced artifacts, not locally reproduced WhiteHole results.

### TwoRooms factors-of-variation scaffold

- **Upstream:** [HackTheWorlds/eb_jepa](https://github.com/HackTheWorlds/eb_jepa)
- **Local repository:** `/Volumes/SSD/repos/worldmodels/HackTheWorlds__eb_jepa`
- **Verified local commit:** `1f8f3a60d3460737de1a6aca6071c9d339c3238f`
- **Relevant path:** `examples/factors_of_variation/`
- **Status:** Local scaffold, not runnable as-is: `main.py:build_jepa()` raises `NotImplementedError`
- **Provides:** severity-sweep configuration and aggregation for dot blur, wall width, and door width.
- **Best reuse:** sweep/evaluation/plotting structure after completing or replacing the model assembly. Blur, geometry, and planning difficulty must remain separate benchmark families.

### Gray–Scott latent PDE rollouts

- **Upstream:** [HackTheWorlds/result-jepadormi](https://github.com/HackTheWorlds/result-jepadormi)
- **Local repository:** `/Volumes/SSD/repos/worldmodels/HackTheWorlds__result-jepadormi`
- **Verified local commit:** `5ff03f87d2d5203239fcde9e126088199106a600`
- **Relevant path:** `examples/gray_scott/`
- **Data:** [polymathic-ai/gray_scott_reaction_diffusion](https://huggingface.co/datasets/polymathic-ai/gray_scott_reaction_diffusion), from [The Well](https://github.com/PolymathicAI/the_well)
- **Status:** Local code; no local JEPA checkpoint found
- **Provides:** six physical regimes, latent and decoded autoregressive rollouts, per-horizon field metrics, OOD parameter evaluation, probes, U-Net/FNO comparisons, and visualizations.
- **Best reuse:** test whether identical PDE fields retain predictive behavior under colormap, contrast, gamma, and background changes. Retraining or checkpoint recovery is required.

### Frozen maze planning

- **Upstream:** [SabaMG/eb_jepa](https://github.com/SabaMG/eb_jepa)
- **Local repository:** `/Volumes/SSD/repos/worldmodels/SabaMG__eb_jepa`
- **Verified local commit:** `bec6e0d91fb7b4ca6c8fa8d10273e6ac39d0c1ed`
- **Relevant path:** `examples/ac_video_jepa/maze/`
- **Status:** Local code; no local checkpoint found
- **Provides:** frozen world-model lookahead, learned subgoals, held-out mazes, success/SPL evaluation, action traces, and GIF output.
- **Best reuse:** evaluate whether appearance-induced representation changes alter action selection or planning despite stable probes. It is secondary because checkpoints must be recovered or models retrained.

## Official model implementations and controls

### I-JEPA

- **Paper:** [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243)
- **Upstream:** [facebookresearch/ijepa](https://github.com/facebookresearch/ijepa)
- **Local repository:** `/Volumes/SSD/repos/ijepa`, commit `52c1ae9`
- **Local checkpoint:** `/Volumes/SSD/huggingface/models/ijepa/IN1K-vit.h.14-300e.pth.tar`
- **Checkpoint origin:** [official ViT-H/14 download](https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar)
- **Status:** Local + artifact
- **Best reuse:** image-side layerwise appearance/content analysis and still-frame analysis of controlled videos.

### V-JEPA 2 and 2.1

- **Paper:** [V-JEPA 2](https://arxiv.org/abs/2506.09985)
- **Upstream:** [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)
- **Local repository:** `/Volumes/SSD/repos/vjepa2`, commit `204698b`
- **Official weights:** [Hugging Face collection](https://huggingface.co/collections/facebook/v-jepa-2-6841bad8413014e185b497a6); the repository also lists direct Meta downloads, including [V-JEPA 2.1 ViT-B distilled from ViT-G at 384 px](https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt) and [V-JEPA 2.1 ViT-G target](https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt)
- **Status:** Local code; the prior V-JEPA 2.1 weights were stored on the deleted GPU pod and must be downloaded again
- **Provides:** intermediate encoder features, attentive probes, action anticipation, action-conditioned prediction, and official evaluation configurations.
- **Best reuse:** principal frozen video-JEPA model and native predictor evaluation.

### LeJEPA and SIGReg

- **Paper:** [LeJEPA](https://arxiv.org/abs/2511.08544)
- **Upstream:** [rbalestr-lab/lejepa](https://github.com/rbalestr-lab/lejepa)
- **Local repository:** `/Volumes/SSD/repos/lejepa`, commit `c293d29`
- **WhiteHole summary:** [`../paper_summaries/summary_2511.08544_lejepa/summary.md`](../paper_summaries/summary_2511.08544_lejepa/summary.md)
- **Compact reference implementation:** `/Users/antonhristov/Documents/ML/AhaWorldModel/docs/paper_summaries/summary_2511.08544_lejepa_sigreg/sigreg_core.py`
- **Status:** Local code
- **Best reuse:** representation-geometry and anti-collapse diagnostics; LeJEPA is not inherently an action-conditioned dynamics model.

### PLDM functional planning

- **Paper:** [Learning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models](https://arxiv.org/abs/2502.14819)
- **Project:** [latent-planning.github.io](https://latent-planning.github.io/)
- **Upstream:** [vladisai/PLDM](https://github.com/vladisai/PLDM)
- **WhiteHole summary:** [`../paper_summaries/summary_2502.14819_pldm/summary.md`](../paper_summaries/summary_2502.14819_pldm/summary.md)
- **Status:** External; no `/Volumes/SSD/repos/PLDM` clone was found
- **Provides:** frozen visual encoder, action-conditioned latent predictor, VICReg/IDM objectives, optional predictor ensemble, MPPI planning, and held-out-layout PointMaze evaluation.
- **Best reuse:** a functional endpoint beyond probes and a controlled setting for rerendering identical layouts/states under appearance changes.

### Non-JEPA controls

- **DINOv2:** [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2), [paper](https://arxiv.org/abs/2304.07193)
- **DINOv3:** [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)
- **VideoMAE-v2:** [OpenGVLab/VideoMAEv2](https://github.com/OpenGVLab/VideoMAEv2), [paper](https://arxiv.org/abs/2303.16727)
- **Status:** External for this inventory; no local clones or weights were verified
- **Best reuse:** distinguish JEPA-family behavior from generic self-supervised ViT behavior. DINOv3's Gram anchoring motivates measuring patch-token Gram drift, while VideoMAE-v2 is the closest standard masked-pixel video control.

## Controlled datasets and benchmark baselines

### SemanticMoments and SimMotion

- **Paper:** [SemanticMoments](https://arxiv.org/abs/2602.09146)
- **Official code:** [saarhub/semantic-moments](https://github.com/saarhub/semantic-moments)
- **Datasets:** [SimMotion-Synthetic](https://huggingface.co/datasets/Shuberman/SimMotion-Synthetic) and [SimMotion-Real](https://huggingface.co/datasets/Shuberman/SimMotion-Real)
- **CVF publication page:** [CVPR 2026 Findings](https://openaccess.thecvf.com/content/CVPR2026F/html/Huberman_SemanticMoments_Training-Free_Motion_Similarity_via_Third_Moment_Features_CVPRF_2026_paper.html)
- **Best reuse:** mandatory retrieval baseline using mean, variance, and third temporal moment features, evaluated at multiple model layers.
- **Limitation:** generated pairs do not certify exact simulator-state equality. Temporal moments are invariant to frame permutation, so reversal, shuffle, and same-states/different-order controls are required.

### Kubric and MOVi

- **Generator:** [google-research/kubric](https://github.com/google-research/kubric)
- **MOVi datasets:** [Kubric MOVi challenge](https://github.com/google-research/kubric/tree/main/challenges/movi)
- **Best reuse:** simulate physics once and rerender the exact state sequence under independent textures, albedos, lighting, backgrounds, and camera-response settings while retaining masks, depth, flow, object states, and contacts.
- **Limitation:** released MOVi datasets are not already organized as same-rollout appearance counterfactuals; custom repeated rendering is required.

### PUG image controls

- **Project:** [PUG](https://pug.metademolab.com/)
- **Paper:** [PUG: Photorealistic and Semantically Controllable Synthetic Data for Representation Learning](https://proceedings.neurips.cc/paper_files/paper/2023/file/8d352fd0f07fde4a74f9476603b3773b-Paper-Datasets_and_Benchmarks.pdf)
- **Best reuse:** image-side texture, environment, and illumination controls. Treat orientation, camera, and object size as geometry rather than appearance.

### Interpreting physics in frozen video models

- **Paper:** [Interpreting Physics in Video World Models](https://arxiv.org/abs/2602.07050)
- **WhiteHole summary:** [`../paper_summaries/summary_2602.07050_interpreting-physics-video-world-models/summary.md`](../paper_summaries/summary_2602.07050_interpreting-physics-video-world-models/summary.md)
- **Best reuse:** layerwise linear/attentive probes, patchwise decoding, subspace geometry, attention suppression, and activation steering protocol.
- **Novelty warning:** it already finds a related intermediate-depth transition in both V-JEPA 2 and VideoMAE-v2, so an intermediate “zone” is not inherently JEPA-specific.

### Worth Remembering novelty baseline

- **Paper:** [Worth Remembering: A Memory for Embodied World Models](https://arxiv.org/abs/2606.03787)
- **Local summary outside WhiteHole:** `/Users/antonhristov/Documents/ML/AhaWorldModel/docs/paper_summaries/summary_2606.03787_worth_remembering/summary.md`
- **Best reuse:** non-predictive latent novelty baseline and residual-memory design reference.
- **Limitation:** surprise-gated memory and residual transplantation are outside the proposed benchmark unless the project returns to memory-based future generation.

## Local conceptual notes

The following verified wiki pages are useful background but are not citable experimental artifacts:

```text
/Users/antonhristov/Documents/wiki/wiki/paper/dinov2-joint-embedding-visual-ssl.md
/Users/antonhristov/Documents/wiki/wiki/paper/paper-ijepa-self-supervised-images-joint-embedding.md
/Users/antonhristov/Documents/wiki/wiki/paper/paper-lejepa-provable-scalable-self-supervised.md
/Users/antonhristov/Documents/wiki/wiki/paper/paper-leworldmodel-stable-end-to-end-jepa-pixels.md
/Users/antonhristov/Documents/wiki/wiki/paper/paper-next-latent-prediction-world-models.md
/Users/antonhristov/Documents/wiki/wiki/document/lejepa-leworldmodel-presentation-notes.md
```

Compact LeWorldModel reference code from the earlier AhaWorldModel project:

```text
/Users/antonhristov/Documents/ML/AhaWorldModel/docs/paper_summaries/summary_2603.19312_leworldmodel/lewm_core.py
```

These compact files explain core algorithms; they are not validated WhiteHole production implementations.

## Resource-to-question map

| Question | First resource | Stronger follow-up |
|---|---|---|
| Does an analysis metric distinguish appearance from physical change? | bounce2D | exact Kubric rerenders |
| Does appearance alter native prediction surprise? | LeWM TwoRooms/Reacher | V-JEPA 2.1 predictor |
| Is content still accessible after its coordinates shift? | I-JEPA/V-JEPA layer probes | source probe versus appearance-specific oracle |
| Does representation drift alter decisions? | LeWM CEM planning | PLDM or maze planning |
| Is the effect specific to JEPA? | VideoMAE-v2, DINOv2/DINOv3 | matched architecture/data/objective training |
| Does a temporal summary actually encode order? | SemanticMoments | reversal, shuffle, and same-state-order controls |

## Evidence boundary

A local repository, checkpoint, figure, or report from another project is a reusable resource, not WhiteHole evidence. Before citing a numerical result as reproduced, WhiteHole must preserve the exact command, environment, checkpoint hash, split, seed, output artifact, and code revision. Public-checkpoint comparisons can describe those model families but cannot isolate the effect of latent prediction from architecture, data, augmentations, scale, or compute.
