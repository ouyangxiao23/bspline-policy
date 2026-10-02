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
