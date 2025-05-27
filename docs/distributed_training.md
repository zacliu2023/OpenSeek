# Distributed Training

## Training on Two Nodes

The baseline configuration is located in the `configs/OpenSeek-Small-v1-Baseline` directory, which includes:

- `config_deepseek_v3_1_4b.yaml`: This is the experiment configuration file, defining the experiment directory, backend engine, task type, and environment settings.

- `train/train_deepseek_v3_1_4b.yaml`: This is the job configuration (job config) file, specifying model parameters, dataset configurations, and training-specific settings.

To customize the experiment configuration, you can modify the baseline configuration directly (or copy it and modify):

- Modify the `experiment.exp_name` field in the experiment configuration file. The experiment output path will be under a directory with this new name.

- Modify the `data.data_path` field in the job configuration file to use your data path and the corresponding data mixing ratios.

Modify the `run_exp.sh` script to specify the directory of your configuration and the path to your experiment configuration file.

For more information, see [Advanced Configuration Guide Link](configs/README.md).

## How to Start a Data Mixing Experiment

The process for conducting a data mixing experiment is as follows:

1. Environment and Tool Installation: Follow the steps in the [Preparation](#preparation) section.

2. Modify Experiment Configuration: Refer to the [Experiment Configuration](#experiment-configuration) section.

3. Start the Experiment: Use the [Running the Baseline](#running-the-baseline) section as a guide for launching your modified script.

4. Check Training Results: Refer to the [Experiment Logs](#experiment-logs) section.

For more details, see [here](openseek/data/data_mix_exp/README.md).