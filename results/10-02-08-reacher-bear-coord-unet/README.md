# Coordinate-aware residual U-Net on dynamic DAVIS bear Reacher

## Result

A 97,731-parameter coordinate-aware residual U-Net beats both unadapted LeWM and the 83,372-parameter STN-only adapter on a screening-excluded 100-case holdout. It also retains source control across two training seeds.

| Condition | Source success | Dynamic bear success |
|---|---:|---:|
| Unadapted LeWM | 73/100 | 19/100 |
| STN-only | 16/100 | 10/100 |
| Coordinate U-Net, seed 123 | 80/100 | 29/100 |
| Coordinate U-Net, seed 124 | **86/100** | **35/100** |

Seed 123 gains 19 bear points over STN-only (95% paired bootstrap CI +8 to +30; exact McNemar p=0.0013). Seed 124 gains 25 points (+14 to +36; p=7.0e-5) and also beats unadapted by 16 points (+4 to +28; p=0.0139). The two-seed means are 83% source and 32% bear success.

This is a **paired-source upper bound**, not a target-only test-time-adaptation result. Training uses exact source renders of each bear-domain simulator state with loss weights:

```text
1 × frozen-dynamics consistency
1 × paired source-latent alignment
10 × paired source-pixel reconstruction
1 × source-latent identity
1 × source-pixel identity
```

The result establishes that a small non-affine input adapter has sufficient capacity to improve this shift when given strong correspondence supervision. It does not yet replace MoVie's reward-free objective.

## Screening and tuning

All screening used the same 30 starts. Target-only dynamics consistency failed, while strong paired pixel reconstruction produced the only useful bear result.

| Adapter/objective | Source | Bear |
|---|---:|---:|
| 83,052-param residual CNN, target dynamics only | 4/30 | 3/30 |
| Residual CNN + source preservation | 24/30 | 3/30 |
| Residual CNN + paired source alignment | 26/30 | 5/30 |
| Coordinate U-Net + paired alignment, pixel weight 1 | 26/30 | 1/30 |
| Coordinate U-Net + paired alignment, pixel weight 10 | 26/30 | **10/30** |

The final holdout excludes all 28 episode IDs encountered during screening and has zero screening-row overlap.

## Protocol

- Training episodes 0-127; 256 training and 64 validation transitions.
- Seeds 123 and 124; 512 optimizer updates each; batch size 32.
- Final evaluation: 100 matched starts, seed 4242, 25-step goals, 50-action budget, CEM horizon 5, 300 samples, 30 iterations, top-k 30.
- Frozen LeWM encoder, projector, dynamics, action encoder, and control core.
- Dynamic `bear_raw_24fps.mp4` background and the dataset/checkpoint versions from the preceding STN experiment.

## Artifacts

- `summary.json`: consolidated metrics, paired tests, hashes, screening split, and limitations.
- `seed123/`, `seed124/`: final training reports and logs.
- `final_holdout/`: per-case outcomes for unadapted, STN-only, and both U-Net seeds.
- `screening/`: failed and successful tuning rows.
- `visuals/`: adapter-output diagnostics.
- `commands.sh`: exact final training and evaluation commands.

Checkpoints are preserved outside Git under `/Volumes/SSD/whitehole/artifacts/10-02-08-reacher-bear-coord-unet/experiment/`.
