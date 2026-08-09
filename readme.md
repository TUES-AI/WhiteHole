# WhiteHole

WhiteHole is a private research playground for experimenting with frozen latent world models under visual observation shifts.

The repository is intentionally exploratory. It is a place for small experiments between collaborators; no result here is currently presented as a final benchmark or paper ablation.

## Current tracks

- **Two Rooms** — a synthetic two-channel environment, offline datasets, JEPA-style source-model training, and input/latent adapter experiments.
- **Reacher** — visual-shift adapter experiments around the external LeWM Reacher model and dataset.
- **Push-T** — a small integration smoke test around an external DINO-WM setup.

The paired Two-Room and Reacher shifts are controlled calibration experiments. They should not be interpreted as general domain-adaptation results.

## Repository layout

```text
configs/          active Two-Room and adapter configurations
scripts/          training, evaluation, visualization, and diagnostics
whitehole/        model, training, adaptation, and planning code
whitehole_envs/   Two-Room environment and dataset code
tests/            lightweight environment tests
results/          committed smoke artifacts and notes
```

## Setup

The core environment is installed with:

```bash
pip install -r requirements.txt
export PYTHONPATH="$PWD"
```

Reacher uses a separate compatible environment because its external solver stack is version-sensitive:

```bash
pip install -r requirements-reacher.txt
```

The Push-T smoke integration uses the external DINO-WM checkout and environment configured in its result notes.

## Useful commands

Generate a small Two-Room dataset and inspect source-model training:

```bash
python whitehole_envs/wall/generate_data.py --help
python -m whitehole.train --help
```

Inspect the Two-Room adaptation entry points:

```bash
PYTHONPATH=. python scripts/train_input_film_adapter.py --help
PYTHONPATH=. python scripts/eval_jepa_baseline.py --help
```

Inspect Reacher tooling:

```bash
PYTHONPATH=. python scripts/visualize_reacher_shifts.py --help
PYTHONPATH=. python scripts/train_reacher_medium_conv_adapter.py --help
PYTHONPATH=. python scripts/eval_reacher_shifts.py --help
```

## Results

Committed Push-T material is an intentionally tiny smoke run. It records that the external stack executed; it is not a benchmark.

No final adaptation conclusions are recorded in this repository yet.

## Development checks

```bash
python -m compileall -q whitehole whitehole_envs scripts tests
python -m pytest -q tests/test_two_rooms_env.py
```
