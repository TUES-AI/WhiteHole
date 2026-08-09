# LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels

**Paper:** Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, and Randall Balestriero, [arXiv:2603.19312v3](https://arxiv.org/abs/2603.19312v3), 2026.  
**Primary material read:** the complete v3-era TeX tree (`neurips_2026.tex`, all four section inputs, `appendix.tex`, equations, algorithms, tables, and figure labels) and the [official implementation](https://github.com/lucas-maes/le-wm) at commit [`8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`](https://github.com/lucas-maes/le-wm/tree/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac). The paper, website, and code are author-controlled sources; the quantitative claims below have not been independently reproduced in WhiteHole.

## Problem and core idea

A next-latent prediction loss alone admits the trivial JEPA solution in which every observation has the same embedding. Existing pixel world models avoid this with frozen pretrained encoders (DINO-WM), stop-gradient/EMA targets, privileged signals, or objectives such as PLDM's prediction, spatial VICReg, temporal VICReg, and inverse-dynamics terms.

LeWorldModel (LeWM) instead jointly trains the pixel encoder and action-conditioned predictor with only

\[
\mathcal L_{\mathrm{LeWM}}
= \underbrace{\|\hat z_{t+1}-z_{t+1}\|_2^2}_{\mathcal L_{\mathrm{pred}}}
+ \lambda\,\underbrace{\mathrm{SIGReg}(Z)}_{\text{anti-collapse}}.
\]

SIGReg pushes the encoder embeddings toward an isotropic Gaussian. It samples unit directions, projects the embeddings onto each direction, and penalizes the discrepancy between each projected empirical characteristic function and that of \(\mathcal N(0,1)\) using an Epps–Pulley statistic. Cramér–Wold motivates the construction: matching *all* one-dimensional projections characterizes the joint distribution. In practice LeWM uses a finite sketch, so this is an anti-collapse pressure rather than a guarantee that finite-batch optimization cannot collapse.

The important WhiteHole distinction is that this paper addresses **source world-model training**, not post-training adaptation under observation shift. It supplies a simpler candidate source loss and useful architecture/planning controls, but no evidence that Gaussianizing target embeddings aligns them to a frozen source predictor.

## Method details (short)

- **Encoder:** a randomly initialized ViT-Tiny (about 5M parameters), patch size 14, 12 layers, 3 heads, hidden/embedding dimension 192. The final `[CLS]` token passes through a projector with BatchNorm. The authors say this projector is needed because the ViT's final LayerNorm interferes with optimizing SIGReg. The released projector is `Linear(192,2048) → BN → GELU → Linear(2048,192)`, not merely a normalization layer ([model config](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/config/train/model/lewm.yaml)).
- **Predictor:** a roughly 10M-parameter, 6-layer causal transformer with 16 heads and 0.1 dropout. An action MLP produces a 192-D condition; zero-initialized AdaLN gates inject it at every transformer block. A second BatchNorm MLP projects predictor outputs into the target space ([implementation](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/module.py#L88-L112)). There is no stop-gradient, EMA target encoder, decoder loss, reward, or proprioception in LeWM.
- **SIGReg:** the paper's default is \(M=1024\) random projections and nominally \(\lambda=0.1\); the released training config uses \(\lambda=0.09\), 1024 projections, and 17 quadrature knots ([loss code](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/module.py#L10-L36), [training objective](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/train.py#L17-L41)). SIGReg is applied step-wise: each time index is tested across the batch, then statistics are averaged over time and projections.
- **Training data:** 224×224 frames, batch size 128, four-frame sub-trajectories, and frame skip 5; five raw actions are concatenated into each action block. The appendix says history length 3 for Push-T/OGBench-Cube and 1 for Two-Room, although the checked-out public config defaults to 3 for all datasets. All reported environment models are described as trained for 10 epochs.
- **Planning:** encode current and goal images, autoregressively roll out candidate action blocks, and minimize terminal latent squared distance \(\|\hat z_H-z_g\|_2^2\). CEM samples 300 plans per iteration and retains 30 elites. Horizon 5 with action block 5 means 25 environment actions of lookahead. The entire five-block plan is executed before replanning (receding horizon 5), so this is relatively open-loop between observations. Push-T uses up to 30 CEM iterations; the implementation-details text says other environments use 10.

## Key results

### Goal-conditioned planning

Every bar below is the paper's labeled success rate. Evaluation uses dataset-derived start/goal tasks: the start is sampled from an offline trajectory and the reachable goal is the state 25 environment steps later in that same trajectory; the action budget is 50. Thus these are short-horizon, in-distribution replay goals, not arbitrary or OOD goals. The source specifies 50 trajectories for the Push-T training-seed ablation, but does not state a general evaluation task count. The paper does not define the error bars in the main figure.

| Environment | LeWM (pixels) | PLDM (pixels) | DINO-WM (pixels) | DINO-WM + proprioception |
|---|---:|---:|---:|---:|
| Two-Room | 87 | 97 | 100 | 100 |
| Reacher | 86 | 78 | 79 | — |
| Push-T | **96** | 78 | 74 | 92 |
| OGBench-Cube | 74 | 65 | **86** | — |

The claimed Push-T improvement over PLDM is **18 percentage points** (96 versus 78), although the prose calls it “18% higher.” LeWM is strongest on Push-T and Reacher, but not uniformly best: it trails PLDM by 10 points and DINO-WM by 13 points on Two-Room, and trails DINO-WM by 12 points on OGBench-Cube. The authors hypothesize that Two-Room's low intrinsic dimension and diversity conflict with a high-dimensional Gaussian prior, but provide no intervention that isolates this explanation.

Under the paper's fixed-FLOP planning comparison, the plotted success rates are LeWM 90 versus DINO-WM 13 on Push-T, and 74 versus 48 on OGBench-Cube. The source does not state the exact FLOP budget. Full planning averaged over 50 runs takes 0.98 s for LeWM versus 47 s for DINO-WM (about 48×), attributed to LeWM using roughly 200× fewer visual tokens; hardware and timing variance are not reported.

### Stability and design ablations

All values in this subsection are Push-T success rates.

- Across three training seeds evaluated on the same 50 trajectories: LeWM \(96.0\pm2.83\), DINO-WM \(92.0\pm1.63\), and PLDM \(78.0\pm5.0\). The caption says “mean … and corresponding variance,” but does not define whether the displayed ± quantity is variance, standard deviation, or another statistic. Three seeds are limited evidence for a broad stability claim.
- Predictor capacity is non-monotonic: Tiny \(80.67\pm6.54\), Small \(96.0\pm2.83\), Base \(86.7\pm3.06\).
- Predictor dropout is consequential: \(p=0\): \(78\pm6.54\); 0.1: \(96.0\pm2.83\); 0.2: \(85.33\pm5.74\); 0.5: \(66.67\pm4.11\). Stable training therefore does not mean architecture choices are immaterial.
- ViT and ResNet-18 encoders obtain \(96.0\pm2.83\) and \(94.0\pm3.27\), respectively. Adding a reconstruction decoder loss reduces performance from \(96.0\pm2.83\) to \(86.0\pm7.54\).
- The plotted success rate stays above 80 for \(\lambda\in[0.01,0.2]\), peaks near 0.09, and falls sharply at 0.5. Projection count and integration-knot plots appear insensitive over the tested ranges, but exact values are not tabulated.
- Solver choice matters substantially. For LeWM: CEM \(96.0\pm2.83\), Adam \(84\pm7.12\), RMSProp \(67.33\pm2.49\), and SGD \(26\pm4.32\). For PLDM: CEM \(78.0\pm5.0\), Adam \(80\pm3.27\), RMSProp \(49.33\pm8.26\), and SGD \(4.67\pm0.06\). These are not stated to be fixed-compute comparisons.

### What the latent analyses do and do not show

- On Two-Room, LeWM and PLDM have the same reported linear position-probe MSE/correlation (LeWM \(0.008\pm0.018, r=0.996\); PLDM \(0.008\pm0.041, r=0.996\)) despite planning rates of 87 and 97. This supports the narrower conclusion that a good one-frame state probe does not guarantee good dynamics or planning.
- On Push-T, LeWM improves over PLDM on all listed probes. For block location, its MLP probe is \(0.001\pm0.006\) MSE with \(r=0.999\), versus PLDM \(0.011\pm0.066\), \(r=0.994\). DINO-WM remains better on several linear or rotational probes.
- OGBench-Cube exposes a weakness in compact end-to-end latents: LeWM's MLP joint-velocity correlation is 0.386 versus DINO-WM's 0.852, and all methods have weak block-yaw correlation (LeWM 0.164, PLDM 0.106, DINO-WM 0.304). Decoded long rollouts likewise lose end-effector orientation.
- Teleport perturbations cause a reported surprise increase with paired-test \(p<0.01\) in Two-Room, Push-T, and Cube, while color changes are weaker and non-significant. Sample counts, effect sizes, and multiple-testing treatment are absent, and teleportation versus color is not magnitude-matched. This detects discontinuities, but by itself does not establish general “physical understanding.”

## What is relevant for WhiteHole

1. **A direct simplification target for source training.** WhiteHole's current Two-Room baseline uses an IMPALA encoder, 512-D GRU predictor, final/tied LayerNorm, and a VICReg + temporal + IDM objective ([config](../../../configs/two_rooms_baseline_jepa.yaml), [VICReg implementation](../../../whitehole/objectives/vicreg.py)). LeWM motivates testing prediction + SIGReg while retaining WhiteHole's architecture, rather than assuming the entire ViT/AdaLN stack is required.
2. **Projector placement is likely material.** WhiteHole's final LayerNorm and LeWM's explicit claim that SIGReg needs a BatchNorm projector create a concrete integration hazard: applying SIGReg directly to the existing normalized 512-D representation is not equivalent to applying it after LeWM's trainable projector.
3. **The paper is not adapter evidence.** WhiteHole has both pre-encoder input adapters ([input affine/residual-conv script](../../../scripts/train_input_film_adapter.py)) and post-encoder target-to-source latent adapters ([adapter module](../../../whitehole/adaptation/adapters.py)). LeWM tests neither. A Gaussian marginal cannot identify source coordinates: any orthogonal rotation preserves \(\mathcal N(0,I)\) while potentially making the frozen predictor unusable. Dynamics or paired alignment remains necessary.
4. **Planning is an essential endpoint.** The Two-Room probe/planning mismatch and the large solver ablation argue against selecting adapters only by paired latent MSE, probe RMSE, or one-step prediction. WhiteHole already exposes terminal representation objectives with MPPI/SGD ([MPC code](../../../whitehole/planning/mpc.py)); CEM is a scientifically useful solver control.
5. **Low-dimensional Two-Room is a stress case, not an easy confirmation.** It is exactly where LeWM underperforms despite excellent position probes. WhiteHole should not assume SIGReg will improve its simplest environment merely because it improves Push-T.

## Concrete experiments to run next

**These are WhiteHole proposals, not experiments reported by the paper.**

1. **Isolate the loss on the existing Two-Room model.** Keep the current IMPALA encoder, 512-D GRU predictor, data, optimizer budget, and planning evaluation fixed. Compare current VICReg+IDM against prediction+SIGReg with \(\lambda\in\{0.01,0.09,0.2\}\), over at least three seeds. Apply SIGReg after a trainable BN projector and include a “directly on final-LN latent” placement control. Report effective rank/per-dimension variance, prediction-to-persistence ratio by horizon, source-domain planning success/cross-wall rate, and seed failures.
2. **Separate LeWM predictor changes from its loss.** In a 2×2 ablation, cross the current GRU versus a causal action-conditioned transformer with current VICReg+IDM versus SIGReg. Within the transformer arm compare zero-initialized AdaLN action injection to action concatenation, holding parameter count approximately fixed; test dropout 0 and 0.1 because the paper's Push-T ablation differs by 18 points between those settings. Evaluate shuffled-action sensitivity and multi-step rollout error before planning.
3. **Adapter-placement ablation under the paired medium shift.** Freeze one source model and compare equal-budget adapters at (a) input pixels before the encoder, (b) after the encoder/final normalization (the current latent adapter), and (c) after the new SIGReg projector but before the frozen predictor. Apply each adapter consistently to current and goal observations. Use identical paired/source-rollout/self-rollout losses and report unadapted/adapted paired MSE, rollout-to-source error by horizon, source-probe transfer, latent active dimensions, and downstream Two-Room MPC success.
4. **Test SIGReg as an adapter regularizer, not as alignment.** For a SIGReg-trained source model, compare current source variance/covariance matching, SIGReg on adapted target latents, and both, while retaining the same paired/dynamics losses. Include a SIGReg-only negative control: it should expose the orthogonal-coordinate ambiguity if marginal normality improves without frozen-predictor rollout or planning improvement.
5. **Fixed-rollout-budget downstream planning comparison.** Add CEM as a control against WhiteHole's MPPI and gradient planner on source, shifted-unadapted, and each adapted representation. Match total model rollouts, action horizon, and replanning frequency rather than solver iterations; sweep “execute all 5 blocks” versus replan every block. Report success, cross-wall rate, final goal error, wall violations, wall-clock latency, and performance versus rollout budget.

## Risks / open questions

- **Finite SIGReg is not the stated asymptotic result.** Cramér–Wold requires all projections; training uses 1024 newly sampled directions, finite batches, and finite quadrature. The theorem does not prove optimizer convergence or rule out every finite-sample collapse mode.
- **The source offers limited stability evidence.** Smooth loss curves and three Push-T seeds do not establish stability across datasets, optimizers, architectures, or low-diversity settings. Two-Room is already a negative downstream case.
- **Evaluation coverage is narrow.** Goals are reachable states 25 steps ahead on the same offline trajectory, horizon is five action blocks, and only 50 tasks are evaluated. The paper itself lists short horizons and dependence on sufficiently diverse offline coverage as limitations.
- **Several reproducibility details are ambiguous or version-dependent.** The appendix describes SIGReg quadrature generically over \([0.2,4]\), while released code integrates over \([0,3]\); prose defaults to \(\lambda=0.1\), code to 0.09; the appendix says Two-Room history 1, current public config uses 3; reported models train for 10 epochs, while the released default config specifies 100. The paper's pseudocode also contains a malformed `F.mse_loss` call, whereas the released training code implements the intended squared difference.
- **Timing and uncertainty reporting are incomplete.** The 47 s versus 0.98 s comparison omits hardware and timing dispersion, main-figure error bars are undefined, and the ± statistic in the three-seed table is ambiguously called “variance.”
- **Gaussian source latents may complicate adaptation.** Distribution matching can erase collapse without preserving predictor-compatible axes, and forcing a high-dimensional isotropic prior may be poorly matched to Two-Room's low-dimensional state manifold. WhiteHole needs dynamics and planning measurements to distinguish these failure modes.
