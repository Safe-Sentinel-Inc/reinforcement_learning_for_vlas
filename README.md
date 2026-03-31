# Reinforcement learning for VLAs

> **Note:** We have a more optimized version of this repository internally. Safe Sentinel tested an early version of this release. We have since modified this codebase and if there are any discrepancies, please [reach out to us](https://github.com/Safe-Sentinel-Inc/reinforcement_learning_vla/issues).
>
> **Limitation:** This public release does not include support for realtime chunking to smooth in-frame transitions.

A reinforcement learning pipeline for training **advantage-conditioned robot manipulation policies** using reinforcement learning for VLAs (Vision-Language-Action models). This project extends the [OpenPI (Physical Intelligence)](https://github.com/Physical-Intelligence/openpi) codebase with RL-based data labeling, K-fold value function training, and iterative policy improvement through DAgger-style data collection.

The core idea: instead of treating all demonstration data equally, a learned **value function** scores each timestep to compute per-step advantages, and the policy is then fine-tuned with **advantage conditioning** so that it preferentially imitates high-advantage actions.

> **Cloud training:** To train the base VLA (pi0.5) on RunPod, see the [RunPod Training Guide](docs/RUNPOD_GUIDE.md). Sample training logs from a test run are available in the [`assets/`](assets/) folder.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Pipeline Steps](#pipeline-steps)
  - [1. Data Conversion](#1-data-conversion)
  - [2. Labeling](#2-labeling)
  - [3. Normalization](#3-normalization)
  - [4. Value Function Training (K-Fold)](#4-value-function-training-k-fold)
  - [5. Value Function Labeling (K-Fold)](#5-value-function-labeling-k-fold)
  - [6. Policy Training](#6-policy-training)
  - [7. Serving](#7-serving)
  - [8. Inference](#8-inference)
  - [9. DAgger (Interactive Data Collection)](#9-dagger-interactive-data-collection)
- [Iterative Data Loop](#iterative-data-loop)
- [License](#license)

---

## Project Overview

This project implements a complete reinforcement learning pipeline for VLA-based advantage-conditioned policy training applied to real-world robot manipulation tasks. The pipeline consists of:

1. **Data collection** -- Teleoperated demonstrations are recorded in MCAP format and converted to LeRobot v2 datasets.
2. **Progress labeling** -- Each timestep in every episode is assigned a progress label (0 to 1) based on episode success/failure, then discretized into 200 bins.
3. **K-fold value function training** -- A distributional value function (SigLIP So400m/14 + Gemma 270M + value head) is trained with K-fold cross-validation to predict per-timestep progress values without overfitting.
4. **Advantage computation** -- The trained value functions infer progress values on their held-out folds, from which temporal-difference advantages are computed. Timesteps above a percentile threshold are labeled as `is_good_action`.
5. **Advantage-conditioned policy training** -- The pi\_0 policy is fine-tuned conditioned on the `is_good_action` label, learning to preferentially reproduce high-advantage actions.
6. **Deployment and DAgger** -- The trained policy is served over WebSocket for real-time robot inference. During deployment, a human operator can intervene via DAgger to correct failures, and the correction data is recorded for the next training iteration.

---

## Repository Structure

```
OpenPI-RL/
|-- src/openpi/                        # Core model and training code (upstream OpenPI -- do not modify)
|   |-- models/
|   |   |-- pi0.py                     # pi_0 flow-matching action generation model
|   |   |-- pi0_fast.py                # pi_0-FAST autoregressive action tokenization
|   |   |-- value_function.py          # Distributional value function (SigLIP + Gemma + value head)
|   |   |-- gemma.py                   # Gemma language model backbone
|   |   |-- siglip.py                  # SigLIP vision encoder
|   |   |-- tokenizer.py               # Action tokenizer
|   |   |-- lora.py                    # LoRA adapters
|   |   +-- model.py                   # Model base classes and ModelType enum
|   |-- policies/
|   |   |-- policy.py                  # Policy inference wrapper
|   |   |-- policy_config.py           # Policy configuration registry
|   |   +-- airbot_policy.py           # Airbot-specific policy transforms
|   |-- training/
|   |   |-- config.py                  # TrainConfig, DataConfig, and config registry
|   |   |-- data_loader.py             # LeRobot dataset loading and batching
|   |   |-- checkpoints.py             # Checkpoint save/restore utilities
|   |   |-- optimizer.py               # Optimizer configuration (AdamW, schedule)
|   |   +-- sharding.py                # JAX multi-device sharding
|   |-- serving/
|   |   +-- websocket_policy_server.py # WebSocket server for remote policy inference
|   |-- shared/                        # Shared utilities (normalization, image tools, etc.)
|   +-- transforms.py                  # Data transform pipeline (normalization, repacking)
|
|-- examples/robot/                    # Robot inference and data conversion scripts
|   |-- inference_sync.py              # Synchronous inference (wait per chunk)
|   |-- inference_async.py             # Asynchronous inference with Temporal Chunk Smoothing (TCS)
|   |-- dagger_controller.py           # DAgger human intervention controller
|   |-- inference_recorder.py          # MCAP recorder for inference-time data
|   |-- convert_mcap_to_lerobot.py     # MCAP to LeRobot v2 format converter
|   |-- robot_config.py                # Robot hardware configuration
|   +-- play_operator.py               # Manual teleoperation playback
|
|-- scripts/                           # Training, labeling, evaluation, and utility scripts
|   |-- train.py                       # Main training entry point (policy or VF)
|   |-- serve_policy.py                # Policy serving entry point
|   |-- compute_norm_stats.py          # Dataset normalization statistics
|   |-- add_returns_to_lerobot.py      # Labeling package (progress, VF inference, advantage)
|   |-- evaluate_pi06_offline.py       # Offline evaluation (metrics, plots)
|   |-- extract_pi06_features.py       # Feature extraction utilities
|   |-- cmds/                          # Shell script wrappers for common operations
|   |   |-- convert_mcap.sh
|   |   |-- add_labels.sh
|   |   |-- compute_stats.sh
|   |   |-- vf_kfold_train.sh
|   |   |-- vf_kfold_label.sh
|   |   |-- vf_train.sh
|   |   |-- vf_label.sh
|   |   |-- train_policy.sh
|   |   |-- serve_policy.sh
|   |   |-- infer_sync.sh
|   |   +-- infer_async.sh
|
|-- packages/openpi-client/            # Client library for remote policy inference
|   +-- src/openpi_client/
|       |-- websocket_client_policy.py # WebSocket client for policy queries
|       |-- action_chunk_broker.py     # Action chunk management
|       |-- base_policy.py             # Abstract policy interface
|       |-- image_tools.py             # Image encoding/decoding utilities
|       +-- runtime/                   # Agent runtime framework
|
|-- pyproject.toml                     # Project metadata, dependencies, tool config
|-- uv.lock                            # Locked dependency versions
+-- LICENSE                            # Apache 2.0
```

---

## Setup

### Requirements

- **Python** >= 3.11
- **uv** -- fast Python package manager ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))
- **CUDA 12** -- required for JAX GPU acceleration and PyTorch
- Multi-GPU recommended for K-fold value function training

### Installation

```bash
# Download the source code
git clone <repository-url>
cd Openpi_RL

# Install all project dependencies and workspace packages
uv sync

# Include optional development dependencies (testing, linting, plotting)
uv sync --group dev
```

The `uv sync` command installs all dependencies listed in `pyproject.toml` and `uv.lock`, including the workspace package `openpi-client` and the pinned LeRobot revision.

### Environment Variable

Most scripts expect `HF_LEROBOT_HOME` to point at the local dataset directory. The shell wrappers in `scripts/cmds/` set this automatically:

```bash
HF_LEROBOT_HOME=./lerobot_data
```

---

## Pipeline Steps

All pipeline steps have corresponding shell wrappers in `scripts/cmds/` with clearly documented CONFIG sections at the top of each file. Edit the CONFIG variables before running.

### 1. Data Conversion

Convert raw MCAP teleoperation recordings to the LeRobot v2 dataset format.

```bash
bash scripts/cmds/convert_mcap.sh
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATA_DIR` | `mcap_data/fold_clothes` | Directory containing MCAP files and a `config.py` |
| `RESUME` | `false` | When `true`, append new episodes without rebuilding the full dataset |

The MCAP directory must contain a `config.py` file describing camera topics, joint topics, and task metadata. The converter reads FlatBuffers-encoded MCAP messages and writes LeRobot v2 parquet files with episode metadata.

### 2. Labeling

Compute progress labels (`binned_value`), intervention flags, and K-fold splits from binary episode success/failure annotations.

```bash
bash scripts/cmds/add_labels.sh
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CONFIG` | `mcap_data/fold_clothes/config.py` | Path to the MCAP config file (contains episode success labels) |
| `REPO_ID` | `Fold_clothes` | LeRobot dataset repository ID |
| `NUM_FOLDS` | `3` | Number of K-fold splits for cross-validated VF training |

Progress labeling logic:

- **Successful episodes**: linear ramp from 0 to 1 over the episode length.
- **Failed episodes**: `linspace(0, 1, T_max)[:n]` with the last 200 steps linearly decayed to 0.
- Values are discretized into **200 bins** spanning [0, 1].

When `NUM_FOLDS > 0`, episodes are randomly assigned to folds and a `meta/folds.json` mapping is written to the dataset.

### 3. Normalization

Compute dataset normalization statistics (mean, std) for state and action dimensions.

```bash
bash scripts/cmds/compute_stats.sh
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `VF_CONFIG` | `pi06_rl_vf_airbot_clothes_folding` | Training config name to compute stats for |

Uses a fast path that reads state/action columns directly from parquet files without decoding images, typically completing in seconds.

### 4. Value Function Training (K-Fold)

Train K value functions in parallel, each excluding one fold of the data, to avoid overfitting the VF to its own training data.

```bash
bash scripts/cmds/vf_kfold_train.sh
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `REPO_ID` | `Fold_clothes` | LeRobot dataset repository ID |
| `VF_CONFIG` | `pi06_rl_vf_airbot_clothes_folding` | Value function training config |
| `GPUS` | `(2 3 4 5 6 7)` | List of available GPU IDs |
| `NUM_FOLDS` | `3` | Number of K-fold splits (must match labeling step) |
| `NUM_TRAIN_STEPS` | `20000` | Training steps per fold |
| `GPUS_PER_FOLD` | `2` | Number of consecutive GPUs allocated per fold |
| `EXP_PREFIX` | `kfold` | Experiment name prefix (fold index is appended) |
| `RESUME` | `false` | When `true`, resume from existing checkpoints |

Each fold trains on all data except the held-out fold. The K training jobs run in parallel, each on its own GPU subset. Checkpoints are saved to `checkpoints/<VF_CONFIG>/<EXP_PREFIX>_fold<N>/`.

For quick single-VF experiments (no cross-validation), use `scripts/cmds/vf_train.sh` instead.

### 5. Value Function Labeling (K-Fold)

Use the K trained value functions to infer progress values on their held-out folds, compute advantages, and write the `is_good_action` boolean column to the dataset.

```bash
bash scripts/cmds/vf_kfold_label.sh
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `REPO_ID` | `Fold_clothes` | LeRobot dataset repository ID |
| `VF_CONFIG` | `pi06_rl_vf_airbot_clothes_folding` | Value function config |
| `GPUS` | `(2 3 4 5 6 7)` | Available GPU IDs |
| `NUM_FOLDS` | `3` | Number of K-fold splits |
| `GPUS_PER_FOLD` | `2` | GPUs per fold for inference |
| `EXP_PREFIX` | `kfold` | Must match the training experiment prefix |
| `CHECKPOINT_STEP` | _(auto-detect)_ | Checkpoint step; leave empty to use the latest |
| `BATCH_SIZE` | `48` | Inference batch size |
| `POSITIVE_FRACTION` | `0.3` | Fraction of timesteps labeled as `is_good_action=True` |
| `GAMMA` | `0.98` | Discount factor for advantage computation |
| `VALUES_DIR` | `/tmp/vf_kfold_Fold_clothes` | Temporary directory for per-fold value predictions |

This script runs in three phases:

1. **Phase 2 -- Inference**: Each fold's VF infers progress values on its held-out episodes (parallel).
2. **Phase 3 -- Merge**: Per-fold predictions are merged and advantages are computed.

The advantage formula:

```
reward(t)    = progress(t+1) - progress(t)
baseline     = 1 / mean_episode_length
advantage(t) = sum_{i=0}^{T-t-1} gamma^i * (reward(t+i) - baseline)
```

Timesteps with advantage above the percentile threshold (determined by `POSITIVE_FRACTION`) are labeled `is_good_action = True`.

### 6. Policy Training

Train the advantage-conditioned pi\_0 policy. Requires `is_good_action` labels in the dataset.

```bash
bash scripts/cmds/train_policy.sh
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `POLICY_CONFIG` | `pi06_rl_pretrain_airbot_clothes_folding` | Policy training config name |
| `EXP_NAME` | `policy_iter0` | Experiment name |
| `GPUS` | `2,3,4,5,6,7` | Comma-separated GPU IDs |
| `OVERWRITE` | `true` | When `true`, train from scratch; when `false`, resume |

The policy is conditioned on the `is_good_action` label during training. At inference time, the label is set to `True` so the policy generates high-advantage actions. Checkpoints are saved to `checkpoints/<POLICY_CONFIG>/<EXP_NAME>/`. Training progress is logged to Weights & Biases.

### 7. Serving

Start the trained policy as a WebSocket inference server.

```bash
bash scripts/cmds/serve_policy.sh
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `POLICY_CONFIG` | `pi06_rl_pretrain_airbot_clothes_folding` | Policy config name |
| `CHECKPOINT_DIR` | `checkpoints/.../XXXXX` | Path to a specific checkpoint step directory |
| `PORT` | `8000` | WebSocket server port |

The server loads the model checkpoint and listens for inference requests from the client. Multiple clients can connect simultaneously.

### 8. Inference

Run the policy on the robot. Two modes are available:

#### Synchronous Inference

Waits for each action chunk to complete before requesting the next one. Suitable for validation and debugging.

```bash
bash scripts/cmds/infer_sync.sh
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `HOST` | `127.0.0.1` | Policy server hostname |
| `PORT` | `8000` | Policy server port |
| `PROMPT` | `"Fold clothes"` | Natural language task prompt |
| `CHUNK_SIZE_EXECUTE` | `25` | Number of action steps to execute per chunk |
| `RECORD` | `false` | Enable MCAP data recording |
| `RECORD_DIR` | `./inference_data` | Directory for recorded data |
| `DAGGER` | `false` | Enable DAgger intervention mode |

#### Asynchronous Inference

Inference and execution run in parallel with Temporal Chunk Smoothing (TCS) for real-time performance. Recommended for production deployment.

```bash
bash scripts/cmds/infer_async.sh
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `HOST` | `127.0.0.1` | Policy server hostname |
| `PORT` | `8000` | Policy server port |
| `PROMPT` | `"Fold clothes"` | Natural language task prompt |
| `CHUNK_SIZE_EXECUTE` | `25` | Action steps per chunk |
| `TCS_DROP_MAX` | `12` | Max stale steps to drop for latency compensation |
| `TCS_MIN_OVERLAP` | `8` | Minimum overlap window for blending old/new chunks |
| `INITIAL_ACTION_WAIT_S` | `10.0` | Max wait time (seconds) for the first action chunk at episode start |
| `RECORD` | `false` | Enable MCAP data recording |
| `RECORD_DIR` | `./inference_data` | Directory for recorded data |
| `DAGGER` | `false` | Enable DAgger intervention mode |

TCS (Temporal Chunk Smoothing) handles the overlap between consecutive action chunks: when a new chunk arrives, the first N stale steps are dropped (up to `TCS_DROP_MAX`), and the remaining overlap region is linearly blended from old to new.

### 9. DAgger (Interactive Data Collection)

DAgger (Dataset Aggregation) enables human intervention during policy inference for iterative data collection. When enabled, a human operator can:

1. Press **`i`** to pause policy inference and enter demonstration mode.
2. The leader arms smoothly align to the follower arm positions via cosine interpolation.
3. The human demonstrates recovery actions through leader arm teleoperation.
4. Press **`o`** to resume autonomous policy inference.

The intervention data (including the human corrections) is recorded in MCAP format when `RECORD=true`. This data can then be converted and added to the training dataset for the next iteration.

To enable DAgger, set `DAGGER=true` in either `infer_sync.sh` or `infer_async.sh`. This requires leader arms to be connected.

---

## Iterative Data Loop

The full iterative improvement loop follows this cycle:

**Step-by-step iteration procedure:**

1. **Initial data collection** -- Record teleoperation demonstrations with the robot. Mark each episode as success or failure in the MCAP config.

2. **Convert** -- Run `convert_mcap.sh` to transform MCAP recordings into LeRobot v2 format. Use `RESUME=true` to append new episodes to an existing dataset.

3. **Label** -- Run `add_labels.sh` to compute progress labels and assign K-fold splits. This must be re-run from scratch whenever episodes are added (not incremental).

4. **Normalize** -- Run `compute_stats.sh` to recompute normalization statistics for the updated dataset.

5. **Train value functions** -- Run `vf_kfold_train.sh` to train K value functions with cross-validation.

6. **Label with VF** -- Run `vf_kfold_label.sh` to infer progress values, compute advantages, and write `is_good_action` labels.

7. **Train policy** -- Run `train_policy.sh` to fine-tune the advantage-conditioned policy. Increment the `EXP_NAME` for each iteration (e.g., `policy_iter0`, `policy_iter1`, ...).

8. **Deploy** -- Run `serve_policy.sh` to start the inference server, then `infer_async.sh` (recommended) or `infer_sync.sh` to run the policy on the robot.

9. **DAgger** -- During deployment, enable DAgger (`DAGGER=true`) and data recording (`RECORD=true`). When the policy fails, the human operator intervenes to demonstrate corrections. The recorded data becomes input for the next iteration.

10. **Repeat** -- Convert the new DAgger data (step 2 with `RESUME=true`), re-label, retrain, and redeploy. Each iteration improves the policy by incorporating both the original demonstrations and the targeted corrections from previous deployments.

---

## Acknowledgments

Thank you to the **Physical Intelligence** team for their outstanding work on the OpenPI framework and the pi\_0 model family. This project builds directly on their research and open-source contributions.

- [pi\_0.6\* blog post](https://www.pi.website/blog/pistar06)
- [OpenPI GitHub repository](https://github.com/Physical-Intelligence/openpi)

We also borrow many ideas and implementation details from **Kai0** by OpenDriveLab.

---

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for the full license text.
