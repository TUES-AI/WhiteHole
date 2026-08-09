# LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics

**Paper:** Randall Balestriero and Yann LeCun, arXiv:2511.08544v3 (2025) — [arXiv](https://arxiv.org/abs/2511.08544v3) · [official code](https://github.com/rbalestr-lab/lejepa/tree/c293d291ca87cd4fddee9d3fffe4e914c7272052)

This summary is based on the full v3 TeX source, including the quantitative tables and proofs in the appendix. Reported results below are **paper claims, not WhiteHole results**, unless explicitly labeled as a WhiteHole hypothesis.

## Problem and core idea

A JEPA predictive/invariance loss can be minimized by complete collapse (all observations receive one embedding) or dimensional collapse (embeddings occupy a low-dimensional subspace). Existing image-SSL systems avoid this with combinations of variance/covariance penalties, stop-gradient, EMA teacher–student networks, predictors, normalization, and tuned schedules.

LeJEPA combines two terms:

1. a squared-error prediction/invariance loss that makes embeddings of related views agree; and
2. **Sketched Isotropic Gaussian Regularization (SIGReg)**, which pushes the embedding distribution toward \(\mathcal N(0,I)\).

The paper argues that isotropic embeddings reduce linear-probe bias and variance. For nonlinear radius-kNN and kernel probes, under smoothness, task-prior, and fixed scalar-covariance assumptions, it further argues that the isotropic Gaussian uniquely minimizes a Fisher-information-dependent integrated-bias objective. These are results for the analyzed probe families and assumptions—not a proof that a Gaussian latent is optimal for every downstream task, action-conditioned world model, or planner.

SIGReg is the main proposed mechanism. It tests many random one-dimensional projections of a minibatch against a standard Gaussian using the Epps–Pulley characteristic-function statistic. The paper's claimed contribution is the statistically motivated, scalable use of this mechanism inside a JEPA. Random slicing itself is not new: the paper explicitly relates it to sliced score matching, sliced Wasserstein methods, and MMD.

## Method details (short)

For embeddings \(Z\in\mathbb R^{N\times K}\), sample \(M\) unit directions \(A\in\mathbb R^{K\times M}\) and form \(Y=ZA\). For each projected dimension, compare its empirical characteristic function

\[
\widehat\phi_Y(t)=\frac1N\sum_{n=1}^N e^{itY_n}
\]

with the standard-normal characteristic function \(\phi_G(t)=e^{-t^2/2}\). The Epps–Pulley discrepancy is

\[
N\int |\widehat\phi_Y(t)-\phi_G(t)|^2w(t)\,dt,
\]

using a Gaussian weight and trapezoidal quadrature. SIGReg averages this statistic over directions; directions are resampled each optimization step. The paper proves bounded sample gradients/curvature for the characteristic-function objective and an \(O(1/N)\) minibatch bias. With fixed numbers of slices and quadrature points, cost is linear in sample count and embedding dimension (more explicitly, projection and CF work scale with \(NKM\) and \(NMT\), where \(T\) is the quadrature size).

The final loss is

\[
\mathcal L_{\mathrm{LeJEPA}}
=(1-\lambda)\mathcal L_{\mathrm{pred}}
+\lambda\,\frac1V\sum_{v=1}^V \mathrm{SIGReg}(Z_v).
\]

The image implementation predicts each view from the mean of the global-view embeddings. For action-conditioned WhiteHole dynamics, the analogous predictive term is the existing predictor error \(\|P(z_t,a_t)-z_{t+1}\|^2\); the action-conditioned predictor is semantically necessary and should not be removed merely because LeJEPA removes the image-only anti-collapse predictor.

The manuscript recommends \(\lambda=0.05\), 17 quadrature points, an integration interval of \([-5,5]\), and 1,024 slices as starting points. The current official minimal example instead exploits symmetry over \([0,3]\), uses 17 points and 256 slices; its implementation is visible in [`MINIMAL.md`](https://github.com/rbalestr-lab/lejepa/blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/MINIMAL.md) and the packaged [Epps–Pulley](https://github.com/rbalestr-lab/lejepa/blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/lejepa/univariate/epps_pulley.py) and [slicing](https://github.com/rbalestr-lab/lejepa/blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/lejepa/multivariate/slicing.py) modules.

## Key results

All numbers are reported by the paper and were not independently reproduced here.

- **ImageNet-1K scale:** the main text reports 77.1% online linear-probe accuracy for ViT-L/14 (0.3B parameters) and 78.5% for ConvNeXtV2-H (0.6B). The abstract separately claims 79% frozen linear evaluation for ViT-H/14, but the supplied TeX does not provide a matching detailed result row, so that number is less auditable than the tabulated results.
- **Architecture breadth:** on ImageNet-10, the paper reports frozen linear-probe top-1 accuracy of 91.5–95% across about 50 `timm` models from eight architecture families, after cross-validating learning rate and weight decay.
- **Hyperparameter ablations:** a ViT-L/14 trained on ImageNet-1K for 100 epochs reaches 72.20, 74.15, 74.72, and 74.07% for batch sizes 128, 256, 512, and 1,024 respectively. Across the reported slice, quadrature, view, projector, and register-token grids, runs avoid chance-level collapse, although accuracy still varies by several points.
- **Galaxy10 in-domain pretraining:** after 400 epochs, frozen ResNet-34 features reach 78.17% full-data accuracy versus 67.62% for DINOv2 ViT-S/16 and 71.38% for DINOv3 ViT-S/16 transfer. With full fine-tuning, ResNet-34 reaches 83.28% versus 78.34% and 81.60%. The comparison changes both pretraining data and model family/size, so it supports the viability of in-domain SSL, not a controlled claim that LeJEPA always dominates generic transfer.
- **Transfer:** after ImageNet-1K pretraining for 100 epochs, LeJEPA ViT-L averages 29.55/60.95/79.48% over eight datasets at 1-shot/10-shot/full-data evaluation. Standard I-JEPA ViT-H trained for 300 epochs gives 30.20/60.51/78.50%; I-JEPA+STOP gives 32.05/62.92/80.70%. Thus LeJEPA is competitive and cheaper in this table, but does not uniformly beat every baseline or dataset.
- **Loss as a model-selection signal:** manuscript text reports roughly 85% Spearman correlation between raw LeJEPA training loss and frozen-probe accuracy, rising to nearly 99% after dividing loss by \(\lambda^{0.4}\) over its experiment pool. This is an empirical image-classification relationship, not yet evidence that the same loss predicts WhiteHole rollout or planning quality.
- **Compute:** the appendix reports SIGReg forward/backward timing on a V100 of 0.465 ms for \(N=M=512\) with 16 integration points and 26.37 ms for \(N=32{,}768,M=512\). Embedding dimension is not stated in that timing table, limiting direct extrapolation.

## What is relevant for WhiteHole

WhiteHole studies frozen latent world models under observation shifts, not primarily generic image classification ([repository scope](../../../README.md)). This changes where LeJEPA is applicable.

- **Direct pretraining relevance:** the current Two-Room source model uses action-conditioned prediction plus VICReg-style variance/covariance terms ([baseline config](../../../configs/two_rooms_baseline_jepa.yaml), [VICReg implementation](../../../whitehole/objectives/vicreg.py)). SIGReg is a principled alternative anti-collapse term that tests more than first and second moments. It can be added while retaining the predictor and IDM objective.
- **Limited direct adaptation relevance:** WhiteHole freezes the source encoder/predictor and trains small input or latent adapters. Existing adapter code can match target-adapted and source standard deviations/covariances ([adapter objectives](../../../whitehole/adaptation/adapters.py)); LeJEPA motivates testing a richer characteristic-function discrepancy. However, forcing adapted outputs to \(\mathcal N(0,I)\) is only appropriate if the frozen source latent is already close to that target. A source-relative sliced ECF discrepancy is the more faithful adaptation control because the frozen predictor expects the source latent geometry.
- **Temporal sampling constraint:** the paper assumes independent samples along its \(N\) axis. WhiteHole trajectories contain strongly correlated timesteps. SIGReg should initially treat independent trajectories at a fixed timestep as samples, or explicitly compare fixed-time results with time-flattened results; flattening all \(T\times B\) latents silently violates the paper's sampling premise.
- **Batch regime:** Two-Room currently uses batch size 64, below the paper's recommended image starting point of 128, although the paper claims the \(O(1/N)\) bias was manageable down to 16. This needs validation for 512-dimensional temporal latents rather than assumption by analogy.
- **Frozen-model scope:** the strongest LeJEPA claims concern learning an encoder from scratch. They do not show recovery from observation shift with a frozen encoder or preservation of action-conditioned rollout geometry. Any WhiteHole adaptation benefit is therefore a new hypothesis, not a result established by this paper.

## Concrete experiments to run next

- **Two-Room source-objective replacement.** **WhiteHole hypothesis:** replacing only the baseline VICReg standard-deviation/covariance terms with SIGReg will preserve anti-collapse while improving frozen location probes or planning. Train matched-seed models with (a) current VICReg, (b) prediction+IDM+SIGReg, and (c) prediction+IDM+VICReg+SIGReg. Keep encoder, predictor, data, optimizer, and parameter budget fixed; report source/shifted location RMSE, active dimensions, covariance spectrum, one-/multi-step rollout MSE, and planning success.
- **Raw latent versus projector ablation.** **WhiteHole hypothesis:** applying SIGReg directly to the 512-D world-state latent better matches the paper's downstream-risk argument, while a disposable MLP projector may optimize more easily. Compare raw latent, linear projector, and 2-layer projector with the same output dimension and loss budget; evaluate the raw latent in every case. This addresses an acknowledged implementation/theory gap in [official issue #17](https://github.com/galilai-group/lejepa/issues/17).
- **Source-relative adapter regularization.** **WhiteHole hypothesis:** for frozen Two-Room adaptation, matching projected target-adapted ECFs to projected source ECFs will preserve source geometry better than either per-feature std/cov matching or a fixed \(\mathcal N(0,I)\) target. Compare those three regularizers on the same affine/residual-conv adapter and dynamics losses. Measure paired latent MSE, source-probe-on-adapted-target RMSE, rollout-to-source MSE by horizon, latent spectrum, and planning success.
- **Sampling and cost ablation.** **WhiteHole hypothesis:** resampled slices remain useful with WhiteHole's small batches, but time flattening may give deceptively low loss because correlated frames inflate \(N\). Sweep batch size \(B\in\{16,32,64,128\}\), slices \(M\in\{16,64,256,1024\}\), and sampling mode `{fixed timestep across trajectories, one random timestep per trajectory, flattened time}`. Record wall time/memory, SIGReg variance across batches, latent diagnostics, and downstream metrics.
- **Label-free checkpoint ranking under shift.** **WhiteHole hypothesis:** validation dynamics loss plus source-relative SIGReg can rank adapter checkpoints without target labels. Across adapter families, seeds, and medium/severe shifts, correlate this score with held-out source-probe RMSE, rollout error, and planning success. Report raw Spearman correlation and any fitted rescaling on a separate validation split; do not reuse the paper's \(\lambda^{0.4}\) exponent without re-estimation.

## Risks / open questions

- **Theory-to-world-model gap:** the optimality analysis concerns linear, radius-kNN, and kernel probes under fixed covariance and smooth/random-task assumptions. Control, long-horizon prediction, and planning are not covered.
- **Projector gap:** the official minimal recipe regularizes projector outputs while downstream probes consume backbone outputs. The author confirms that projectors often help empirically but that the reason remains open ([issue #17](https://github.com/galilai-group/lejepa/issues/17)); isotropy of projector outputs does not by itself establish isotropy of WhiteHole's state latent.
- **Finite slices versus the guarantee:** Cramér–Wold identifies a multivariate distribution from all one-dimensional projections. The consistency theorem uses a growing dense direction set and a maximum statistic, while practical SIGReg uses the average over a finite, resampled set. The smoothness/resampling arguments motivate this approximation but do not make every finite-slice optimum uniquely Gaussian.
- **Prediction-loss algebra:** the appendix's proof of the equivalence between its pairwise global-view loss and center-prediction loss uses \(V_g^{-1}\sum_v\|z_v\|^2=\|V_g^{-1}\sum_v z_v\|^2\), which is false unless global views are identical. The missing term is their within-global-view variance. The released code directly implements center prediction, so WhiteHole should treat that as the actual algorithm rather than rely on the stated equivalence.
- **“Without heuristics” is narrow:** LeJEPA removes several anti-collapse devices, but the experiments still use multi-crop augmentation, a projector, AdamW, learning-rate warmup/cosine annealing, and choices for views, slices, and quadrature. The manuscript also gives inconsistent default view counts (2 global + 8 local in one recommendation versus 8 total with 2 global elsewhere), and official code defaults differ from the manuscript.
- **Target mismatch during adaptation:** an already-trained WhiteHole source latent need not be standard Gaussian. Direct Gaussianization can alter scale, neighborhoods, and transition geometry expected by the frozen predictor even while SIGReg improves; source-relative metrics and rollout evaluation are required.
- **Novelty boundary:** SIGReg's JEPA formulation and claimed package of guarantees are the paper's contribution; characteristic-function goodness-of-fit tests, random projections, sliced distribution matching, and Gaussian regularization all have prior art. WhiteHole experiments should describe their use as an adaptation/application, not as a new general distribution-matching principle.
