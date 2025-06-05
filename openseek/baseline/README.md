# OpenSeek Baseline

This directory contains scripts for setting up and running the OpenSeek baseline model experiments. The baseline serves as a standardized reference point for evaluating improvements in model architectures, training techniques, and data processing methods.

## Overview

The OpenSeek baseline consists of:
- A standardized environment setup
- A predefined dataset (OpenSeek-Pretrain-100B)
- A reference model implementation (1.4B parameters with 0.4B active parameters)
- Configuration files located in `configs/OpenSeek-Small-v1-Baseline/`

## Scripts

### 1. `setup.sh`

This script prepares the environment for running the OpenSeek baseline experiments.

#### What it does:
1. Prompts for virtual environment activation
2. Downloads the OpenSeek-Pretrain-100B dataset from Hugging Face
3. Clones the FlagScale repository at a specific commit
4. Runs the unpatch script to prepare the backend

#### Usage:
```bash
bash setup.sh
```

#### Note:
It's recommended to manually download the dataset from https://huggingface.co/datasets/BAAI/OpenSeek-Pretrain-100B and update the `experiment.dataset_base_dir` path in `configs/OpenSeek-Small-v1-Baseline/config_deepseek_v3_1_4b.yaml`.

### 2. `run_exp.sh`

This script manages the baseline model training process.

#### Features:
- Starts, profiles, or stops training sessions
- Supports different configuration options
- Integrates with FlagScale for distributed training

#### Usage:
```bash
# Start baseline training
bash run_exp.sh start

# Profile model performance
bash run_exp.sh profile

# Stop a running training session
bash run_exp.sh stop
```

#### Additional options:
You can specify a different configuration by adding a second parameter:
```bash
bash run_exp.sh start llama
```

## Getting Started

1. Make sure you have the necessary dependencies installed
2. Run the setup script to prepare the environment:
   ```bash
   bash setup.sh
   ```
3. Start the baseline training:
   ```bash
   bash run_exp.sh start
   ```

## Configuration

The baseline uses configuration files located in `configs/OpenSeek-Small-v1-Baseline/`:
- `config_deepseek_v3_1_4b.yaml`: Main experiment configuration
- `train/train_deepseek_v3_1_4b.yaml`: Training-specific configuration

You can modify these files to adjust parameters like:
- Model architecture
- Training hyperparameters
- Data mixture settings

## Verification

To verify your training is running correctly, check:
1. The `OpenSeek-Small-v1-Baseline/logs/` directory for log files
2. Look for training progress in the log file, e.g.:
   ```
   grep "iteration.*consumed samples" OpenSeek-Small-v1-Baseline/logs/host_0_localhost.output
   ```

## References

- [OpenSeek GitHub Repository](https://github.com/FlagAI-Open/OpenSeek)
- [FlagScale Framework](https://github.com/FlagOpen/FlagScale)
- [OpenSeek-Pretrain-100B Dataset](https://huggingface.co/datasets/BAAI/OpenSeek-Pretrain-100B) 