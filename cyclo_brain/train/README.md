# cyclo-train

Config-driven policy training for Cyclo Intelligence. **One YAML per run.**

Everything runs through one command:

```bash
./docker/container.sh train-lerobot <experiment>
```

한국어 문서: **[README_KR.md](README_KR.md)**

---

## Contents

1. [Quick start](#quick-start)
2. [Full workflow example](#full-workflow-example)
3. [Commands](#commands)
4. [Writing an experiment](#writing-an-experiment)
5. [Where files go](#where-files-go)
6. [Weights & Biases](#weights--biases)
7. [Choosing a GPU](#choosing-a-gpu)
8. [What each file is for](#what-each-file-is-for)
9. [Development](#development)

---

## Quick start

```bash
# see the exact trainer command without running anything
./docker/container.sh train-lerobot peanut_act_80k --dry-run

# train (detached tmux session)
./docker/container.sh train-lerobot peanut_act_80k

# watch
tmux attach -t cyclo_train

# list runs
./docker/container.sh train-lerobot --list

# publish a finished run for inference
./docker/container.sh train-lerobot --promote peanut_act_80k
```

A 30-second end-to-end check on a small dataset:

```bash
./docker/container.sh train-lerobot act_smoke --no-tmux --set tracking.backend=none
```

---

## Full workflow example

Training a new ACT policy on a freshly recorded task, start to finish.

### 1. Get the dataset

Datasets come from the HuggingFace Hub. This downloads into
`workspace/dataset/<name>` and converts v2.1 → v3.0 automatically.

```bash
./docker/container.sh train-lerobot download RobotisSW/Task_900010_MyTask_lerobot
```

Output tells you the final path and whether a v2.1 backup was kept:

```
[cyclo-train] dataset: /workspace/dataset/Task_900010_MyTask_lerobot  (v3.0)
[cyclo-train] v2.1 backup kept at: /workspace/dataset/Task_900010_MyTask_lerobot_old
```

### 2. Drop the mobile-base dimensions (if your robot has them)

The FFW-SG2 records 22 DOFs: 19 body joints + 3 mobile-base
(`linear_x`, `linear_y`, `angular_z`). For a stationary task, train on 19:

```bash
./docker/container.sh train-lerobot trim-mobile Task_900010_MyTask_lerobot
```

```
[cyclo-train] trimmed 22 -> 19 dims: /workspace/dataset/Task_900010_MyTask_lerobot_no_mobile
```

The source is left untouched, and the copy uses hardlinks so the videos cost no
extra disk.

> The dimension you train on must match what the inference engine assembles at
> serving time (`lerobot_engine`'s `_state_modalities`). Train on 19 dims and the
> engine must not include the `mobile` modality.

### 3. Write the experiment

`experiments/task_900010_myTask_act.yaml`:

```yaml
_base: _base/act.yaml

name: Task_900010_MyTask_act_80k

dataset:
  repo_id: RobotisSW/Task_900010_MyTask_lerobot
  root: ${workspace}/dataset/Task_900010_MyTask_lerobot_no_mobile
```

### 4. Check it before spending a GPU-day

```bash
./docker/container.sh train-lerobot task_900010_myTask_act --dry-run
```

This resolves `_base`, expands `${workspace}`, validates the dataset exists, and
prints the exact `lerobot-train` command. Nothing is written.

### 5. Train

```bash
./docker/container.sh train-lerobot task_900010_myTask_act
```

```
[cyclo-train] Starting run 'Task_900010_MyTask_act_80k' (lerobot)
[cyclo-train] run dir:  .../docker/workspace/runs/Task_900010_MyTask_act_80k
[cyclo-train] log:      .../runs/Task_900010_MyTask_act_80k/train.log
[cyclo-train] tracking: wandb → cyclo-lerobot
[container.sh] Training started in tmux session: cyclo_train
```

Watch it:

```bash
tmux attach -t cyclo_train                       # Ctrl-b d to detach
tail -f docker/workspace/runs/Task_900010_MyTask_act_80k/train.log
```

### 6. If it stops early, resume

```bash
./docker/container.sh train-lerobot task_900010_myTask_act --resume
```

Picks up from the last checkpoint and continues to the original step count. The
log and manifest of the earlier attempt are preserved.

### 7. Decide when it has converged

```bash
./docker/container.sh train-lerobot analyze-loss Task_900010_MyTask_act_80k --interval 10000
```

```
step       loss       abs_change   rel_change   stable
    10,000   0.512340     -            -            no
    20,000   0.221010   0.291330       56.86%       no
    ...
Recommended step: 60,000
```

### 8. Publish it for inference

```bash
./docker/container.sh train-lerobot --list
./docker/container.sh train-lerobot --promote Task_900010_MyTask_act_80k
# or pick a specific checkpoint:
./docker/container.sh train-lerobot --promote Task_900010_MyTask_act_80k --step 060000
```

The model appears at `workspace/model/lerobot/<name>/`, ready to select on the
Inference page. Upload it to the Hub from the UI (Edit Dataset → Hugging Face)
if you want to share it.

---

## Commands

All are `./docker/container.sh train-lerobot …`.

| Command | What it does |
|---|---|
| `<experiment>` | Start a training run |
| `<experiment> --resume` | Continue from the last checkpoint |
| `<experiment> --dry-run` | Print the resolved trainer command, run nothing |
| `<experiment> --set K=V` | Override any config key (repeatable) |
| `--list` | List runs, their checkpoints, and where they were published |
| `--promote <run>` | Copy a checkpoint into the inference dropbox |
| `download <repo_id>` | Fetch a dataset from the Hub, convert v2.1 → v3.0 |
| `convert <dataset>` | Convert a local v2.1 dataset to v3.0 (`--no-mobile` to trim too) |
| `trim-mobile <dataset>` | Drop the 3 mobile DOFs from a v3.0 dataset |
| `analyze-loss <run>` | Find the loss plateau in a run's log |
| `--help` | Full options |

Useful flags: `--no-tmux` (run in the foreground), `--session NAME` (parallel
runs), `--build` (rebuild the training image).

---

## Writing an experiment

Inherit the pinned defaults and override only what changes:

```yaml
_base: _base/act.yaml

name: my_run                      # names the run dir and the published model; must be unique

dataset:
  repo_id: RobotisSW/Task_900006_..._v30    # identity/provenance
  root: ${workspace}/dataset/Task_900006_..._no_mobile   # what is actually loaded

policy:
  kl_weight: 5.0                  # the only delta from the base
```

- `${workspace}` and `${name}` expand automatically.
- Keys under `policy:` become `--policy.<key>=<value>`; keys under `train:`
  become `--<key>=<value>`. Any lerobot flag is reachable from YAML without
  touching Python.
- To make a variant, add a config with `_base` — don't copy an experiment file.

Override anything for a single run without editing the file:

```bash
./docker/container.sh train-lerobot peanut_act_80k --set train.steps=2000 --set name=quick_probe
```

Write exponents in YAML form: `--set policy.optimizer_lr=1.0e-5`.

---

## Where files go

```
workspace/                                  # git-ignored data
├── dataset/<name>/                         # datasets
├── runs/<name>/                            # one directory per training run
│   ├── manifest.json                       #   what code + data produced this
│   ├── resolved_config.yaml                #   the fully resolved config
│   ├── train.log                           #   full trainer output
│   └── output/checkpoints/{000010,…,last}/ #   checkpoints
└── model/lerobot/<name>/                   # inference dropbox — written only by `promote`
    ├── checkpoints/<step>/pretrained_model/
    └── manifest.json
```

Training writes only to `runs/`. `promote` is the only thing that writes to
`model/`, so only checkpoints you have chosen are selectable for inference. It
copies `pretrained_model/` alone, leaving the optimizer state (hundreds of MB per
checkpoint, needed only to resume) out of what you deploy and upload.

`workspace/` is written by the container as root, so removing a run needs
`sudo rm -rf docker/workspace/runs/<name>`.

### The manifest

Every run records what produced it, before training starts:

| Field | |
|---|---|
| `code.repo_git_sha`, `repo_dirty` | which code |
| `code.lerobot_submodule_sha` | which trainer |
| `code.image` | which image |
| `config_hash`, `config` | which settings |
| `dataset.action_shape`, `observation_state_shape` | 19 vs 22 dims |
| `runtime.gpus`, `torch`, `cuda_visible_devices` | which hardware |
| `argv` | the literal command |
| `resumes[]` | one entry per resume |

It is copied alongside the model when you promote, so a published model always
carries its provenance.

---

## Weights & Biases

```bash
cp docker/.env.example docker/.env      # git-ignored
# add: WANDB_API_KEY=<your key from https://wandb.ai/authorize>
```

Configure per experiment (or in `_base/act.yaml` for everything):

```yaml
tracking:
  backend: wandb          # wandb | none
  project: cyclo-lerobot
  entity: robotis-intern-team-peanut-belt   # must be an entity your account belongs to
  mode: online            # online | offline | disabled
```

The run URL is printed at startup. Metrics are logged every `train.log_freq`
steps (200 by default), so a run shorter than that logs no curve.

If `wandb` is unavailable, the run continues with local logs and says so; pass
`--strict-tracking` to make that a hard error instead. Checkpoints are not
uploaded as wandb artifacts — they are already on disk.

---

## Choosing a GPU

```bash
CUDA_VISIBLE_DEVICES=1 ./docker/container.sh train-lerobot peanut_act_80k
```

Use `CUDA_VISIBLE_DEVICES`, not `policy.device: cuda:1` — the training path uses
HuggingFace `accelerate`, which picks the device itself and only reads
`policy.device` to force CPU. Leave the variable **unset** to use all GPUs; an
empty value hides every GPU.

---

## What each file is for

### Configs

| File | Purpose |
|---|---|
| `experiments/_base/act.yaml` | The pinned ACT hyperparameters. The single copy — every ACT experiment inherits it. |
| `experiments/act_smoke.yaml` | 100-step run on a small dataset; the end-to-end check. |
| `experiments/peanut_act_80k.yaml` | Production peanut pick-and-place run. |
| `experiments/peanut_act_kl5.yaml` | Same, with `kl_weight: 5.0`. |

### Package

| File | Purpose |
|---|---|
| `cyclo_train/cli.py` | Command line: every subcommand lives here. |
| `cyclo_train/config.py` | Loads a YAML, folds in `_base`, expands `${…}`, validates, hashes. Owns every path a run uses. |
| `cyclo_train/backends/base.py` | The `TrainerBackend` contract: turn a config into an argv. Nothing else. |
| `cyclo_train/backends/lerobot.py` | Builds the `lerobot-train` command line. Wraps upstream; never patches it. |
| `cyclo_train/runner.py` | Guards, creates the run dir, writes the manifest, streams the trainer's output to your terminal and the log. |
| `cyclo_train/manifest.py` | Collects provenance (git, image, dataset, GPU) into `manifest.json`. |
| `cyclo_train/promote.py` | Copies a checkpoint into the inference dropbox; lists runs. |
| `cyclo_train/data_prep/dataset.py` | Dataset identity, paths, `info.json`, and the shared 22→19 trim plan. |
| `cyclo_train/data_prep/download.py` | Hub download + upstream v2.1→v3.0 conversion. |
| `cyclo_train/data_prep/convert_v21.py` | The local v2.1→v3.0 converter (can trim in the same pass). |
| `cyclo_train/data_prep/no_mobile.py` | Trims a v3.0 dataset from 22 to 19 dims. |
| `cyclo_train/analysis/loss.py` | Parses a train log and finds the loss plateau. |

### Outside this directory

| File | Purpose |
|---|---|
| `docker/container.sh` | `train-lerobot` — the only host entry point. Starts the container, injects git provenance, manages tmux. |
| `docker/docker-compose.yml` | The `lerobot_train` service (profile `train`, so it never auto-starts). |
| `cyclo_brain/policy/lerobot/Dockerfile.train` | The training image: the serving image plus `wandb`. |
| `docker/.env` | `WANDB_API_KEY` and friends. Git-ignored. |

---

## Development

```bash
cd cyclo_brain/train
python -m pytest tests/ -q          # no GPU, dataset, or container required
```

| Test file | Covers |
|---|---|
| `test_config_and_argv.py` | Config resolution and the generated trainer command. |
| `test_runner_and_promote.py` | Run directory, log, manifest, exit codes, promotion. |
| `test_data_prep_and_analysis.py` | Trim planning, dataset paths, download argv, loss analysis. |
| `test_container_entrypoint.py` | The `container.sh train-lerobot` subcommand. |

Two constraints worth keeping:

- **The package stays pure-Python (`pyyaml` only) at import time.** It is
  bind-mounted read-only into images and must import unchanged in all of them.
  `numpy`/`pandas` are imported inside `data_prep` functions, never at module
  level. A test enforces this.
- **Backends build strings and nothing else** — no I/O, no side effects. That is
  what keeps them testable without a GPU.

### Adding a policy backend

1. Add `cyclo_train/backends/<name>.py` implementing `TrainerBackend`.
2. Add `experiments/_base/<name>.yaml` with its pinned defaults.
3. Import it in `backends/__init__.py` to register it.
