# Official DINO-WM Push-T Smoke Test

Date: 2026-08-07
Cluster: Discoverer
Job: 186307
Status: completed, exit code 0:0
Runtime: 00:00:59

This is a minimal end-to-end smoke test for the official DINO-WM Push-T checkpoint. It verifies that the official checkpoint, official Push-T dataset, DINOv2 encoder loading, Push-T environment, and planner execute on Discoverer.

This is not a benchmark score. The planner settings were intentionally tiny so the job would terminate quickly.

## Environment

Current environment: official DINO-WM Push-T source environment.

- External repo: `/valhalla/projects/bg-eng-01/dino_wm`
- DINO-WM commit checked out on Discoverer: `0a9492f`
- Gym id: `pusht`
- Entry point: `env.pusht.pusht_wrapper:PushTWrapper`
- Dataset: official `pusht_noise`
- Dataset path: `/valhalla/projects/bg-eng-01/dino_wm/official_artifacts/data/pusht_noise`
- Checkpoint: `/valhalla/projects/bg-eng-01/dino_wm/official_artifacts/extracted/outputs/pusht/checkpoints/model_latest.pth`
- Validation rollouts loaded: 21
- Train rollouts loaded: 18685

The environment used here is still the normal/source Push-T appearance:

- Block shape: `T`
- Block color: `LightSlateGray`
- Goal color: `LightGreen`
- Agent color: `RoyalBlue`
- `with_velocity: true`
- `with_target: true`

No shape shift or texture shift has been applied yet.

## Smoke Test Overrides

```text
model_name=pusht
model_epoch=latest
n_evals=1
n_plot_samples=1
goal_source=random_state
goal_H=1
objective.alpha=1
planner.max_iter=1
planner.sub_planner.num_samples=4
planner.sub_planner.topk=2
planner.sub_planner.opt_steps=1
```

## Result

```text
success_rate: 0.0
mean_state_dist: 226.3010594160317
mean_visual_dist: 8.07653725552721
mean_proprio_dist: 12.931950569152832
mean_div_visual_emb: 46.21829605102539
mean_div_proprio_emb: 0.06975600868463516
```

The earlier unbounded smoke job, `186268`, used the original `planner.max_iter: null` behavior and timed out after reaching `MPC iter 220` with success rate 0.0. That timeout confirmed the pipeline ran, but it is not a usable benchmark result.

## Remote Artifacts

The larger rendered videos/images were left on Discoverer:

```text
/valhalla/projects/bg-eng-01/dino_wm/plan_outputs/pusht_official_bound_
```

Only compact text artifacts are stored in this repository.
