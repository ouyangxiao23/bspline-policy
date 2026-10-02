# BSP-DP / RoboCasa TurnOffMicrowave

Method reproduction using the unmodified BSP source at repository root, commit `61ed5f42fced971d50a89b46417493790876ccd1`. `third_party/bsp` is a relative link to that source. Task: press the stop button and move the gripper away.

The [paper](https://arxiv.org/html/2607.09648v1), Table 2(a), reports dense DP 77% and BSP-DP 89%. These are reference numbers, not our measured results. Exact paper data/environment versions and task training budget are unspecified.

## Training

Official human image dataset: 54 demonstrations, 11827 frames, three 128x128 camera views, 12-dimensional delta actions. Source and hash are in `configs/reproduction.json`. Reindexing preserves trajectory contents and repairs references to a missing demonstration. Shared seed-42 split: 49 training and 5 validation demonstrations.

Both models use official datasets, normalizers, UNet, DDIM and workspace implementations. Batch 64, FP32, 601 epochs, learning rate 1e-4, warmup 500 updates, cosine decay, EMA; random 116x116 crops retained. BSP uses degree 3, 16 parameter rows and fitting tolerance 0.002. Dense DP executes 8 actions per chunk. Batch 8 outputs from the initial run remain archived on the training node.

## Usage

Use a Python environment compatible with the upstream dependencies. Set `EVAL_PYTHON` to its executable and optionally `MICROWAVE_DEPS` (default `/tmp/bsp-microwave-deps`). OpenGL system libraries must be available. Then, from this directory:

```bash
mkdir -p data reports
"$EVAL_PYTHON" scripts/setup_node_dependencies.py
export PYTHONPATH="${MICROWAVE_DEPS:-/tmp/bsp-microwave-deps}:${PYTHONPATH:-}"
# Download the source_url recorded in configs/reproduction.json to:
# data/turn_off_microwave_human_im.hdf5
"$EVAL_PYTHON" scripts/reindex_dataset.py
"$EVAL_PYTHON" scripts/inspect_hdf5.py data/turn_off_microwave_human_im_indexed.hdf5 > reports/dataset_inspection.json
"$EVAL_PYTHON" scripts/configure_training.py
"$EVAL_PYTHON" scripts/launch_pair.py
```

The launcher checks both models before training on GPUs 0 and 2. `MICROWAVE_DATASET` can override the indexed dataset path. Logs and status are local; data, checkpoints and runtime caches are excluded from Git. Included reports capture batch-64 smoke checks and dataset provenance; absolute paths describe the original node.

## Evaluation status

Closed-loop rollout has not yet been validated. The dataset records robosuite 1.4.1 and an old RoboCasa protocol; the candidate RoboCasa 1.0.1 environment is not verified as compatible. Spline boundary decoding and control frequency must also be checked before reporting success rates. `scripts/audit_sources.py` expects the separately pinned RoboCasa candidate checkout at `third_party/robocasa`.

Current training settings are in `actual_training` in reproduction.json; proposed settings and pending items describe earlier planning, not the active run.

## Interim rollout

The old mobile robot environment is pinned separately by `scripts/prepare_eval_sources.py`. Install MuJoCo 3.1.1 into an isolated `MICROWAVE_EVAL_DEPS` target (default `/tmp/bsp-microwave-eval-deps`); keep the training environment unchanged. Run `scripts/prepare_eval_assets.py` to download the task assets (Objaverse only, matching the task default).

Place stable, independent checkpoint copies at `outputs/rollout_mid/checkpoints/{dense,bsp}.ckpt`; do not evaluate files while saving. `scripts/launch_rollout_mid.py` checks paired reset state hashes, both policy forward passes and successful replay of validation demo 4, then evaluates 50 seeds per policy across four shards on GPUs 0/2. It uses EMA, 20 Hz control, 500 steps and an execution cap of 8 actions.

Spline knot repair follows the upstream deployment `safer_knots` rule; sampling uses frame-index times with the current history frame at time 1, clamped to the valid half-open domain to avoid padded endpoint artifacts. This is an explicit simulation adapter, not a verified paper evaluation protocol. Video is recorded for the first two episodes.

## Measured interim result (epoch 380)

| Model | Successes / valid tests | Success rate | Paper Table 2(a) |
| --- | --- | --- | --- |
| Dense DP | 31 / 50 | 62% | 77% |
| BSP-DP | 16 / 50 | 32% | 89% |

All 50 paired initial state hashes match. Seed 100021 initially satisfied success before any policy action; both models exclude it and use seed 100050 instead. No runtime failures remain. Detailed episodes, Wilson confidence intervals and protocol limits are in `reports/rollout_interim_comparison.json`. This intermediate, single-seed, 20 Hz result does not reproduce the paper's reported BSP benefit or its stated 100 Hz sampling protocol. Final training is still in progress.

## Final checkpoint evaluation

Both training processes completed 601 epochs successfully (zero-based checkpoint epoch 600). Place stable independent final copies at `outputs/rollout_final/checkpoints/{dense,bsp}.ckpt` and launch `scripts/launch_rollout_final.py`. It reuses the 50 eligible interim seeds, runs four shards per model, and verifies both policy pairing and the interim/final initial-state hashes before final summary. Final results are stored separately from epoch 380.

## Final measured result

Both policies completed the same 50 valid paired tests at checkpoint epoch 600 (601 training epochs): dense DP 36/50 (72%), BSP-DP 10/50 (20%). At epoch 380 they achieved 31/50 (62%) and 16/50 (32%), respectively. No runtime failures occurred; initial-state hashes match across models and checkpoints. Results and protocol limits are in `reports/rollout_final_comparison.json`. The current protocol does not reproduce the paper's BSP benefit.
