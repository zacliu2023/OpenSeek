<div align="center">
  <img src="./openseek_logo.jpg" alt="OpenSeek Logo" width="150">

</div>

<div align="center">

OpenSeek is dedicated to uniting the global open-source community to drive collaborative innovation in algorithms, data, and systems, with the goal of developing next-generation models that surpass DeepSeek.

English| [简体中文](README_zh.md)


[![GitHub license](https://img.shields.io/github/license/FlagAI-Open/OpenSeek)](https://github.com/FlagAI-Open/OpenSeek/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/FlagAI-Open/OpenSeek)](https://github.com/FlagAI-Open/OpenSeek/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/FlagAI-Open/OpenSeek)](https://github.com/FlagAI-Open/OpenSeek/network)
[![GitHub issues](https://img.shields.io/github/issues/FlagAI-Open/OpenSeek)](https://github.com/FlagAI-Open/OpenSeek/issues)

</div>

# 📌 Project Overview
OpenSeek is an open source project initiated by the Beijing Academy of Artificial Intelligence (BAAI), aiming to unite the global open source communities to drive collaborative innovation in algorithms, data and systems to develop next-generation models that surpass DeepSeek. Drawing inspiration from large model initiatives like Bigscience and OPT, the project is dedicated to building an independent open source algorithmic innovation system. Since the open sourcing of the DeepSeek model, academia has seen numerous algorithmic improvements and breakthroughs, but these innovations often lack complete code implementations, necessary computational resources, and high-quality data support. The OpenSeek project hopes to explore high-quality dataset construction mechanisms through uniting the open source community, promote open sourcing of the entire large model training pipeline, build innovative training and inference code to support various AI chips besides Nvidia, and promote independent technological innovation and application development.

**Objectives of OpenSeek:**
- Innovative data synthesis technology: Address the challenge of acquiring high-quality data and break through data barriers.
- Support for multiple AI chips: Reduce dependency on specific chips and improve model universality and adaptability.
- Build an independent open source algorithmic innovation system: Promote independent algorithmic innovation and technology sharing through open source collaboration.

**Project:** https://github.com/orgs/FlagAI-Open/projects/1 

**Acknowledgments & Contribution Guidelines**

We extend our sincere gratitude to the FlagScale team for their foundational framework support. This project is built upon FlagScale's robust infrastructure.

- *For framework-related discussions/issues*
Please direct your questions and report framework-specific issues through FlagScale's GitHub Issues. Code contributions should be submitted via Pull Requests (PRs) to the FlagScale repository.

- *For data strategies & training methodologies*
Discussions, proposals, and PRs regarding dataset implementations, training optimizations, and experimental configurations should be initiated through this project's GitHub Issues and Pull Requests.



# 📢 News
- 🔥[05/06/2025] **Data group**-release bilingual pretrainning dataset CCI4.0-M2-V1 <u>*[[readme](Docs/README_CCI4.0_M2_V1.md)]*</u>, **Algo group**-release the pretrained model OpenSeek-Small V1 <u>*[[readme](Docs/README_OPENSEEK_SMALL_V1.md)][[download](Docs/OpenSeek-Small_V1_download_link)]*.</u>
- 🔥[03/20/2025] #4 online meetup 19:00-20:00 : https://meeting.tencent.com/crm/NL4rAjg489
- 🔥[03/20/2025] #3 online meetup 19:00-20:00 ：https://meeting.tencent.com/crm/NXwDAyLG59
- 🔥[03/06/2025] #2 online meetup 19:00-20:00 ：https://meeting.tencent.com/crm/2pxo8BBDb7
- 🔥[02/25/2025] #1 online meetup 18:00-19:00 ：https://meeting.tencent.com/v2/cloud-record/share?id=e188482b-0105-43f9-b8e7-cf5f1e4d136b&from=3&is-single=false&record_type=2
- 🔥[02/13/2025] Completed experiment on the OpenSeek-PT-1T dataset, released model checkpoints, data ratios, training codes with hyperparameters, and wandb logs.

# 🚗 Getting Started
## Preparation
1. Clone this repository:
```shell
git clone https://github.com/FlagAI-Open/OpenSeek.git path/to/OpenSeek
```
2. Install the [FlagScale](https://github.com/FlagOpen/FlagScale) tool (skip this step if already installed):
- Using Docker (Recommended):
```shell

```
- From Source:
```shell
# Clone the repository
git clone https://github.com/FlagOpen/FlagScale.git path/to/FlagScale

# Install the requirements
cd path/to/FlagScale/install
./install-requirements.sh --env train
./install-requirements.sh --env inference

# Install the packages with customized extensions
cd vllm
pip install .

pip install -e ./megatron-energon
cp -r megatron-energon/src/megatron/energon megatron/megatron
```
- For more details, see [FlagScale](https://github.com/FlagOpen/FlagScale) or [readme](docs/FlagScale_Usage.md).

3. Download the [OpenSeek-Pretrain-100B](https://huggingface.co/datasets/BAAI/OpenSeek-Pretrain-100B) dataset from Huggingface.

## Running the Baseline
1. Copy the `openseek/algorithm/run_exp.sh` script to the FlagScale root directory:
```shell
cp path/to/OpenSeek/openseek/algorithm/run_exp.sh path/to/FlagScale
```

2. Modify the configuration path in `run_exp.sh`. Change line 30 to
```shell
python3 run.py --config-path path/to/OpenSeek/configs/OpenSeek-Small-v1-Baseline --config-name $2
```

3. Navigate to the FlagScale root directory and run the baseline:
```shell
cd path/to/FlagScale
bash run_exp.sh start config_deepseek_v3_1_4b.yaml
```

4. For detailed explanations of the experiment configuration, see [Experiment Configuration](#experiment-configuration).

## Experiment Logs

Experiment logs will be output to the Aquila-1_4B-A0_4B-Baseline/logs directory. For example, when training with two machines, each having eight GPUs:

- The experiment startup log is located in the path corresponding to the first GPU on the first host. For example:

```
Aquila-1_4B-A0_4B-Baseline/logs/details/host_0_xxx.xxx.xxx.xxx/20250423_185352.022338/default_atongk86/attempt_0/0
```

- The training loss log is located in the path corresponding to the last GPU on the last host. For example:

```
Aquila-1_4B-A0_4B-Baseline/logs/details/host_1_yyy.yyy.yyy.yyy/20250423_185352.918040/default_zcuhq1c7/attempt_0/7
```

# 📚 Data

## CCI4.0-M2 v1

[CCI4.0-M2 V1](docs/README_CCI4.0_M2_V1.md) is a comprehensive multilingual dataset suite designed to support various stages of large language model training. It is composed of three targeted subsets, each serving a distinct purpose:

|| CCI4.0-M2-Base v1 | CCI4.0-M2-CoT v1 | CCI4.0-M2-Extra v1 |
|--|--|--|--|
|Huggingface| https://huggingface.co/datasets/BAAI/CCI4.0-M2-Base-v1 | https://huggingface.co/datasets/BAAI/CCI4.0-M2-CoT-v1 | https://huggingface.co/datasets/BAAI/CCI4.0-M2-Extra-v1 |
|Notes|This is the core pretraining subset, aimed at building general-purpose language understanding. It contains approximately 30% Chinese (from both collaborative and open-source sources) and 70% English (mainly from Nemotron-CC), with all data sourced from web pages.|This subset focuses on enhancing the model’s reasoning abilities through synthesized Chain-of-Thought (CoT) data. It provides step-by-step reasoning trajectories generated from various data sources, enabling improved performance on complex inference tasks.|Designed as a supplement to the core training data, this subset offers domain-specific knowledge to improve the model’s performance in specialized fields.|

Together, these three components make CCI4.0-M2 v1 a well-rounded and scalable dataset foundation for training advanced language models across general, reasoning, and domain-specific tasks.

In addition to the main suite, [OpenSeek-Pretrain-100B](docs/100B_pipeline.md) was randomly sampled from the CCI4.0-M2 v1 datasets. This 100B data subset is specifically used for experimental training purposes.

Your can find more details about data [here](docs/Data.md).


# 🚀 Training

## Stage 1


| | OpenSeek-Small-v1-Baseline | OpenSeek-Small-v1 | OpenSeek-Mid-v1 |
|--|--|--|--|
|Parameter size| 1.4B (0.4B active) | 1.4B (0.4B active) | 16B (3B active) |
|Number of tokens|100B|720B|200B|
|Checkpoint|https://huggingface.co/BAAI/OpenSeek-Small-v1-Baseline|https://huggingface.co/BAAI/OpenSeek-Small-v1|https://huggingface.co/BAAI/OpenSeek-Mid-v1|
|Training config|[config_deepseek_v3_1_4b.yaml](configs/OpenSeek-Small-v1-Baseline/config_deepseek_v3_1_4b.yaml) [train_deepseek_v3_3b_1330B.yaml](configs/OpenSeek-Small-v1/train_deepseek_v3_3b_1330B.yaml)|[config_deepseek_v3_3b_1330B.yaml](configs/OpenSeek-Small-v1/config_deepseek_v3_3b_1330B.yaml) [train_deepseek_v3_3b_1330B.yaml](configs/OpenSeek-Small-v1/train_deepseek_v3_3b_1330B.yaml)|[config_deepseek_v3_16b.yaml](configs/OpenSeek-Mid-v1/config_deepseek_v3_16b.yaml) [train_deepseek_v3_16b.yaml](configs/OpenSeek-Mid-v1/train_deepseek_v3_16b.yaml)|
|Notes|We sampled 100 billion tokens from the CCI4.0 dataset and trained a 1.4B-parameter MoE model with 0.4B active parameters. This model, along with the dataset, is open-sourced as a baseline for future experiments in areas such as dataset construction, algorithmic strategies, and parallel training frameworks.|OpenSeek-Small v1 is the first-stage production model from the OpenSeek project, designed as a foundation for next-generation language models. The model is trained based on the CCI4.0 dataset, with a total training volume of 720 billion tokens. Among them, approximately 10% are Chinese and 60% are English, while the remaining data comes from various sources such as books, academic papers, encyclopedias, mathematics, code, and synthetic data.|The OpenSeek-Mid-v1 model adopts a fine-grained coefficient MoE architecture with 16 billion total parameters and 3 billion active parameters, similar to the structures of DeepSeek-V2-Lite and Moonlight-16B-A3B. Based on this architecture, we conducted experiments on training hyperparameters, the Multiple Token Predictor submodule, and data mixing ratios. Under the final training configuration, the model was trained on 200 billion tokens, and the resulting weights are released as a temporary and experimental checkpoint.|

# 🖥️ System
TODO

# 🔋 Advanced Usage

## Experiment Configuration

The baseline configuration is located in the `configs/OpenSeek-Small-v1-Baseline` directory, which includes:

- `config_deepseek_v3_1_4b.yaml`: This is the experiment configuration file, defining the experiment directory, backend engine, task type, and environment settings.

- `train/train_deepseek_v3_1_4b.yaml`: This is the job configuration (job config) file, specifying model parameters, dataset configurations, and training-specific settings.

To customize the experiment configuration, you can modify the baseline configuration directly (or copy it and modify):

- Modify the `experiment.exp_name` field in the experiment configuration file. The experiment output path will be under a directory with this new name.

- Modify the `data.data_path` field in the job configuration file to use your data path and the corresponding data mixing ratios.

Modify the `run_exp.sh` script to specify the directory of your configuration and the path to your experiment configuration file.

For more information, see [Advanced Configuration Guide Link](configs/README.md).

## Data Mixing Experiment

The process for conducting a data mixing experiment is as follows:

1. Environment and Tool Installation: Follow the steps in the [Preparation](#preparation) section.

2. Modify Experiment Configuration: Refer to the [Experiment Configuration](#experiment-configuration) section.

3. Start the Experiment: Use the [Running the Baseline](#running-the-baseline) section as a guide for launching your modified script.

4. Check Training Results: Refer to the [Experiment Logs](#experiment-logs) section.

For more details, see [here](openseek/data/data_mix_exp/README.md).
## Training Experiment

TODO

# 👁 Project Highlights
- *High-Quality Data Accessibility:*
Open-source 10TB-level, high-quality Chinese and English pretraining data, ensuring robust and diverse model training resources.
- *Scalable Data Synthesis Strategy:*
A streamlined and scalable approach to synthesizing Chain-of-Thought (CoT) data, leveraging Webpage, Code, Math, Wiki, and Book sources to enhance reasoning capabilities.
- *Multi-AI Chip Support:*
  Built on Triton, the project offers optimized support for multiple AI chips, ensuring flexibility and adaptability across diverse hardware ecosystems.
- *High-Performance Training Infrastructure:*
  Highly optimized training support, designed to maximize efficiency and accelerate model development.
- *Advanced Model Architecture:*
  A more efficient model structure, optimized for performance and scalability, enabling superior computational efficiency and inference speed.


# ☎️ Open-Source Co-construction Plan
OpenSeek thrives on community collaboration. We believe in the collective intelligence of developers worldwide and welcome contributions that advance this project toward excellence.

For detailed information on how to contribute, please refer to our [Contribution Guide](CONTRIBUTING.md).

Together, we can explore the frontiers of large language models and drive technological innovation through open source collaboration.

<div align="center">
  <img src="./wechat.png" alt="wechat" width="200">
</div>

# 📜 License Agreement
- Apache 2.0

