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
OpenSeek is an open source project initiated by the Beijing Academy of Artificial Intelligence (BAAI), aiming to unite the global open source communities to drive collaborative innovation in algorithms, data and systems to develop next-generation models that surpass DeepSeek. Drawing inspiration from large model initiatives like Bigscience and OPT, the project is dedicated to building an independent open source algorithmic innovation system. Since the open sourcing of the DeepSeek model, academia has seen numerous algorithmic improvements and breakthroughs, but these innovations often lack complete code implementations, necessary computational resources, and high-quality data support. The OpenSeek project aims to explore high-quality dataset construction mechanisms through uniting the open source community, promote open sourcing of the entire large model training pipeline, build innovative training and inference code to support various AI chips besides Nvidia, and promote independent technological innovation and application development.

**Objectives of OpenSeek:**
- **Advanced data technology**: Address the challenge of acquiring high-quality data.
- **Multiple AI devices support**: Reduce dependency on specific chips and improve model universality and adaptability.
- **Standalised LLM training baseline**: Promote independent algorithmic innovation and technology sharing through open source collaboration.

**Project:** https://github.com/orgs/FlagAI-Open/projects/1 

**Acknowledgments & Contribution Guidelines**

Thanks to FlagScale team for their support for OpenSeek Training. 

- *For system-related improvements*
Please report framework-specific issues to [FlagScale's GitHub Issues](https://github.com/FlagOpen/FlagScale/issues). Code contributions should be submitted via Pull Requests (PRs) to the [FlagScale](https://github.com/FlagOpen/FlagScale).

- *For data & algorithm improvements*
Discussions of dataset implementations, training optimizations, and experimental configurations in [here](https://github.com/FlagAI-Open/OpenSeek/issues).



# 📢 News
- 🔥[05/06/2025] **Data group**-release bilingual pretrainning dataset CCI4.0-M2-V1 <u>*[[readme](Docs/README_CCI4.0_M2_V1.md)]*</u>, **Algo group**-release the pretrained model OpenSeek-Small V1 <u>*[[readme](Docs/README_OPENSEEK_SMALL_V1.md)][[download](Docs/OpenSeek-Small_V1_download_link)]*.</u>
- 🔥[03/20/2025] #4 online meetup 19:00-20:00 :  [[screen recording]](https://meeting.tencent.com/crm/NL4rAjg489)
- 🔥[03/20/2025] #3 online meetup 19:00-20:00 ：[[screen recording]](https://meeting.tencent.com/crm/NXwDAyLG59)
- 🔥[03/06/2025] #2 online meetup 19:00-20:00 ：[[screen recording]](https://meeting.tencent.com/crm/2pxo8BBDb7)
- 🔥[02/25/2025] #1 online meetup 18:00-19:00 ：[[screen recording]](https://meeting.tencent.com/v2/cloud-record/share?id=e188482b-0105-43f9-b8e7-cf5f1e4d136b&from=3&is-single=false&record_type=2)
- 🔥[02/13/2025] Completed experiments on OpenSeek-PT-1T dataset, [more]().

# 🚗 Getting Started
## Preparation
1. Clone this repository and enter the directory:
```shell
git clone https://github.com/FlagAI-Open/OpenSeek.git
cd OpenSeek
```
2. Install the [FlagScale](https://github.com/FlagOpen/FlagScale) tool (or move it to OpenSeek if you have installed it somewhere else):

- From Source:
```shell
# Clone the repository
git clone https://github.com/FlagOpen/FlagScale.git

# Install the requirements
cd FlagScale/install
./install-requirements.sh --env train
```

- Using Docker (coming soon)

- For more details, see [FlagScale](https://github.com/FlagOpen/FlagScale) or [readme](docs/FlagScale_Usage.md).

3. Download the [OpenSeek-Pretrain-100B](https://huggingface.co/datasets/BAAI/OpenSeek-Pretrain-100B) dataset to OpenSeek.

**You can also run the following script to build up your project environment after you have built python environment and activated it:**

```
bash openseek/baseline/setup.sh
```

## Running the Baseline
Make sure you have completed the environment installation and configuration as outlined in the [previous section](#preparation). Next, you can run the baseline with a simple command:
```shell
bash openseek/baseline/run_exp.sh start
```

## Where is the Log?

Experiment logs will be output to the `Aquila-1_4B-A0_4B-Baseline/logs` directory. For example, when training with two machines, each having eight GPUs:

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

[CCI4.0-M2 V1](docs/README_CCI4.0_M2_V1.md) is a large-scale bilingual pre-training dataset engineered for superior data quality and diverse human-like reasoning trajector:

|| CCI4.0-M2-Base v1 | CCI4.0-M2-CoT v1 |
|--|--|--|
|Download| [huggingface](https://huggingface.co/datasets/BAAI/CCI4.0-M2-Base-v1) | [huggingface](https://huggingface.co/datasets/BAAI/CCI4.0-M2-CoT-v1) |
|Notes| 5.2TB Chinese webpage, 22TB English webpage, some data released in [CCI4.0-M2-Extra](https://huggingface.co/datasets/BAAI/CCI4.0-M2-Extra-v1) due to the license concern. | 45 million CoT sample covers math, code, arxiv, wiki and webpage|





In addition to the main suite, [OpenSeek-Pretrain-100B](docs/100B_pipeline.md) was randomly sampled from the CCI4.0-M2 v1 datasets. This 100B data subset is specifically used for experimental training purposes.

Your can find more details about data [here](docs/Data.md).


# 🚀 Algorithm

## Stage 1


| | OpenSeek-Small-v1-Baseline | OpenSeek-Small-v1 | OpenSeek-Mid-v1 |
|--|--|--|--|
|Parameter size| 1.4B (0.4B active) | 1.4B (0.4B active) | 16B (3B active) |
|Number of tokens|100B|720B|200B|
|Checkpoint|[huggingface](https://huggingface.co/BAAI/OpenSeek-Small-v1-Baseline)|[huggingface](https://huggingface.co/BAAI/OpenSeek-Small-v1)|[huggingface](https://huggingface.co/BAAI/OpenSeek-Mid-v1)|
|Wandb|[wandb](https://wandb.ai/aquila3/OpenSeek-3B-v0.1/runs/Aquila-1_4B-A0_4B-Baseline-rank-31)|[wandb](https://wandb.ai/aquila3/Aquila-1_4B-A0_4B-1330B)|[wandb](https://wandb.ai/aquila3/OpenSeek-3B-v0.1/runs/DeepSeek-V3-16B3A-K81-dist02-rank-2047)|
|Evaluation|[evaluation](https://huggingface.co/BAAI/OpenSeek-Small-v1-Baseline#evalation)|[evaluation](https://huggingface.co/BAAI/OpenSeek-Small-v1#benchmark-performance)|[evaluation](https://huggingface.co/BAAI/OpenSeek-Mid-v1/blob/main/README.md#evalation)|
|Experiment Config|[Experiment Config](configs/OpenSeek-Small-v1-Baseline/config_deepseek_v3_1_4b.yaml)|[Experiment Config](configs/OpenSeek-Small-v1/config_deepseek_v3_3b_1330B.yaml)|[Experiment Config](configs/OpenSeek-Mid-v1/config_deepseek_v3_16b.yaml) |
|Training config| [Training Config](configs/OpenSeek-Small-v1-Baseline/train/train_deepseek_v3_1_4b.yaml)|[Training Config](configs/OpenSeek-Small-v1/train/train_deepseek_v3_3b_1330B.yaml)|[Training Config](configs/OpenSeek-Mid-v1/train/train_deepseek_v3_16b.yaml)|
|Notes|This model is open-sourced as a baseline for future experiments in areas such as dataset construction, algorithmic strategies, and parallel training frameworks.|OpenSeek-Small v1 is the first-stage production model from the OpenSeek project, designed as a foundation for next-generation language models. |We conducted experiments on training hyperparameters, the Multiple Token Predictor submodule, and data mixing ratios. The resulting weights are released as a temporary and experimental checkpoint.|

> The usage and difference of Experiment Config and Training Config are explained [here](#experiment-configuration).

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

## How to Start a Data Mixing Experiment

The process for conducting a data mixing experiment is as follows:

1. Environment and Tool Installation: Follow the steps in the [Preparation](#preparation) section.

2. Modify Experiment Configuration: Refer to the [Experiment Configuration](#experiment-configuration) section.

3. Start the Experiment: Use the [Running the Baseline](#running-the-baseline) section as a guide for launching your modified script.

4. Check Training Results: Refer to the [Experiment Logs](#experiment-logs) section.

For more details, see [here](openseek/data/data_mix_exp/README.md).

## How to Train on Multiple Machines
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

