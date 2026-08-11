# Frozen JEPA visual-adapter screen

One-seed exploratory screen of the same framewise image adapters on frozen I-JEPA and V-JEPA 2.1. Large adapter checkpoints are stored outside Git at `/Volumes/SSD/whitehole/artifacts/11-jepa-visual-adapters/`.

## Protocol

- Seeds: I-JEPA `123`; V-JEPA `20260811`.
- Adapters: 97,731-parameter coordinate-aware residual U-Net and 67,009-parameter grid/color rectifier.
- I-JEPA: official ViT-H/14 ImageNet-1K checkpoint; 20-class Imagenette + Imagewoof; disjoint probe/adaptation/source-prior/capacity/final subsets of 2,000/2,000/2,000/400/1,000 images.
- Fixed target domains: RGB to RBG; +30 degree rotation with +18/+8 degree x/y shear and reflection padding; their composition.
- V-JEPA: official V-JEPA 2.1 ViT-B student/predictor with the official ViT-G target encoder; five balanced `nateraw/kinetics-mini` classes; disjoint probe/adaptation/source-prior/capacity/final subsets of 25/15/10/25/25 videos. Each clip has 16 frames sampled at 4 FPS and center-cropped to 384 pixels. RBG is identical on all frames.
- Capacity gate: paired pixel training must recover at least 50% of the separate capacity-validation pixel gap before target-only or target-plus-source-identity rows run.

## Primary results

I-JEPA baseline frozen-probe accuracy was 95.0% on source, 92.7% on RBG, 90.4% on fixed affine, and 78.3% on fixed composed images. Paired RBG passed for both adapters; target-only masked-I-JEPA and target masked-I-JEPA plus a disjoint clean-source identity term stayed at the 92.7% unadapted target accuracy. Neither adapter passed paired capacity for affine or composed shifts, so their non-paired rows were skipped.

V-JEPA's tiny five-class probe scored 88% on both source and RBG final videos, so this dataset did not expose a downstream RBG gap. Within each class, lexicographically sorted filenames were partitioned without random shuffling, making this tiny split ordering-dependent. Paired U-Net and grid/color training recovered only 29.1% and 38.1% of the capacity-validation pixel gap, below the gate; target-only and target-plus-source-identity rows were skipped. The U-Net retained 88% source accuracy but reduced target accuracy to 84%; grid/color remained at 88%/88%. This is an inconclusive downstream video benchmark and a negative capacity result for these adapters/budgets. V-JEPA losses use one final clip from each class rather than one contiguous class block.

## Safety and execution checks

- Real-checkpoint I-JEPA MPS and CUDA updates propagated finite nonzero adapter gradients with zero frozen-core gradient tensors and zero sentinel drift. CUDA peak allocation was 2,845,008,384 bytes. The MPS command was not preserved and is only a supplementary interactive safety record; the CUDA stdout is archived.
- A faithful V-JEPA target-loss update using the ViT-B context encoder/predictor and ViT-G target encoder completed in 3.29 seconds. Loss was `0.5852251`, gradient norm `0.2162652`, maximum update `9.999998e-05`, frozen-core gradient tensors zero, and peak allocation 12,875,506,176 bytes. Its exact command and output are preserved in `commands.sh` and `vjepa-target-gradient-smoke.log`; it is separate from the paired-update `vjepa-cuda-smoke.log`.
- Final I-JEPA and V-JEPA frozen-core sentinel drift was zero.

## Checkpoints and hashes

- I-JEPA `IN1K-vit.h.14-300e.pth.tar`: `0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22`
- V-JEPA ViT-B student: `848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d`
- V-JEPA ViT-G teacher: `7aae1a3c7a31d258af9c985388b5d2f20587469380f2e11b54d7876ac8cfe58a`
- Fixed I-JEPA report: `9ccacea3a9bbcd193a71cd0d072f7085641a9dabb0826fac7136eb9696b61ecc`
- V-JEPA report: `7be65b705b3f707f3f5bd6f044d4afc41758a7f25f26b46c7f95f1a76a794b81`

Checkpoint copies were verified against all 22 remote SHA-256 values before GPU deletion.

## Environment and commands

GPU: NVIDIA RTX A6000, 49,140 MiB, driver 570.133.20. Python 3.13.14; PyTorch 2.11.0+cu128; torchvision 0.26.0+cu128; PyAV 18.0.0; timm 1.0.28. Official repositories were I-JEPA `52c1ae9` and V-JEPA 2 `204698b`.

Exact primary I-JEPA and V-JEPA invocations are in `commands.sh`; the training logs contain program output rather than shell command echoes.

## Files

- `ijepa-fixed-report.json`: primary fixed-domain aggregate and row metrics.
- `ijepa-variable-affine-report.json`: non-primary developmental record. Its pre-fix code snapshot and command were not preserved, so it is retained only as a non-reproducible protocol failure and its metrics are not evidence.
- `vjepa-report.json`: five-class RBG video screen.
- `*-train.log`, `*-cuda-smoke.log`, and `*-gradient-smoke.json`: execution records.
- `ijepa-fixed-shift-examples.png`: source and fixed-shift examples.
