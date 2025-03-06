<div align="center">
  <img src="./openseek_logo.jpg" alt="OpenSeek Logo" width="150">

</div>

<div align="center">

OpenSeek aims to unite the global open source community to drive collaborative innovation in algorithms, data and systems to develop next-generation models that surpass DeepSeek.
English| [简体中文](README_zh.md)

[![GitHub license](https://img.shields.io/github/license/FlagAI-Open/OpenSeek)](https://github.com/FlagAI-Open/OpenSeek/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/FlagAI-Open/OpenSeek)](https://github.com/FlagAI-Open/OpenSeek/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/FlagAI-Open/OpenSeek)](https://github.com/FlagAI-Open/OpenSeek/network)
[![GitHub issues](https://img.shields.io/github/issues/FlagAI-Open/OpenSeek)](https://github.com/FlagAI-Open/OpenSeek/issues)

</div>

# 📌 Project Overview
OpenSeek is an open source project initiated by the Beijing Academy of Artificial Intelligence (BAAI), aiming to unite the global open source communities to drive collaborative innovation in algorithms, data and systems to develop next-generation models that surpass DeepSeek. Drawing inspiration from large model initiatives like Bigscience and OPT, the project is dedicated to building an independent open source algorithmic innovation system. Since the open sourcing of the DeepSeek model, academia has seen numerous algorithmic improvements and breakthroughs, but these innovations often lack complete code implementations, necessary computational resources, and high-quality data support. The OpenSeek project hopes to explore high-quality dataset construction mechanisms through uniting the open source community, promote open sourcing of the entire large model training pipeline, build innovative training and inference code to support various AI chips besides Nvidia, and promote independent technological innovation and application development.

**Core Objectives of OpenSeek:**
- Innovative data synthesis technology: Address the challenge of acquiring high-quality data and break through data barriers.
- Support for multiple AI chips: Reduce dependency on specific chips and improve model universality and adaptability.
- Build an independent open source algorithmic innovation system: Promote independent algorithmic innovation and technology sharing through open source collaboration.

**Project Repository:** https://github.com/FlagAI-Open/OpenSeek

# 📢 News
- 🔥[02/25/2025] #1 online meetup 18:00-19:00 ：https://meeting.tencent.com/v2/cloud-record/share?id=e188482b-0105-43f9-b8e7-cf5f1e4d136b&from=3&is-single=false&record_type=2
- 🔥[02/13/2025] Completed validation of the OpenSeek-PT-1T dataset on a 3B size model, released model checkpoints, data ratios, training codes with hyperparameters, and wandb logs.


# 👁 Project Highlights
 - High-quality data open and accessible
  - Open source large-scale high-quality Chinese and English datasets (>4TB), covering a wide variety of data types and scenarios.
  - Open source high-quality dataset construction plans, supporting diverse high-quality data synthesis based on human data, helping developers achieve innovation at the data level.
- Multi-AI chip distributed training framework
  - Support for Triton operators, multi-chip training, compatible with various hardware architectures, ensuring efficient utilization of different devices.
  - Implement more efficient computation, communication, and memory access collaborative hybrid parallel schemes, providing cluster training logs and performance data to help developers optimize large-scale training tasks.
- Model structure optimization and improvement
  - Explore optimization of two different model sizes, OpenSeek-small and OpenSeek-Mid, to meet the needs of different application scenarios.
  - Provide training experiences and optimization plans for small-sized models to help developers achieve high-performance development and deployment in resource-constrained environments.

# ☎️ Open Source Co-construction Plan
As a member of the open source community, we deeply understand that the power of open source comes from the wisdom and enthusiasm of every developer. We firmly believe that through the joint efforts of global developers, every contribution will push the project towards maturity and perfection.

Welcome to check our [Contribution Guide](CONTRIBUTING.md) for more details.

Whether you are:
- A deep learning expert with experience in large model training;
- A data scientist dedicated to data construction and algorithm innovation;
- Or a beginner passionate about open source projects;

You can find a platform to showcase your talents at OpenSeek. You can contribute in the following ways:
- Code and technical solution contributions
  - If you have unique insights into training processes, code implementation, or performance optimization, feel free to submit a Pull Request and help us advance the project.
- Data, algorithm, and resource support
  - If you have high-quality datasets, innovative algorithms, or other valuable resources and wish to contribute in non-code forms, please contact us directly to discuss collaboration methods.
- Participate in technical discussions and documentation improvement
  - Share your insights, experiences, and suggestions to help us continuously improve project documentation and technical details.
- Synthetic Reasoning Data Co-construction Plan
  - We will update the [Huggingface](https://huggingface.co/datasets/BAAI/OpenSeek-Synthetic-Reasoning-Data-Examples) platform with sample data, effects, and strategies used for each version of our synthesized samples (if the strategies provided by external contributors we will cite thanks to involved team or individual). Currently recommended optimization directions for reasoning data synthesis include:
    - Building a domain labeling system to balance data diversity:
      - Labeling system in Code Domain
      - Labeling system in Math Domain
      - Labeling system in Paper (Arxiv) Domain
      - Labeling system in Wiki Domain
      - Labeling system in Webpage Domain
    - Synthetic data quality evaluation and screening:
      - Synthetic Data Quality Evaluation and Screening in Code Domain
      - Synthetic Data Quality Assessment and Screening in Math Damain
      - Synthetic Data Quality Assessment and Screening in Paper (Arxiv) Domain
      - Synthetic Data Quality Assessment and Screening in Wiki Domain
      - Synthetic Data Quality Assessment and Screening in Webpage Domain
    - Synthesis pipeline optimization:
      - Synthesis Pipeline Optimization in Code Domain
      - Synthetic Pipeline Optimization in Math Domain
      - Synthesis Pipeline Optimization for the Paper (Arxiv) Domain
      - Synthetic Pipeline Optimization for the Wiki Domain
      - Synthesis Pipeline Optimization for the Webpage Domain
- Foundational Pretraining Data Plan
  - We will continuously update our datasets, performance results, and data construction methodologies on Hugging Face for each version across multiple domains. If the construction methodology is contributed by external collaborators, we will acknowledge and credit the respective teams or individuals.The recommended directions for dataset iteration include:

    - Building High-Quality Web Datasets
      - Manual validation of dataset sample quality
      - Benchmark-based Decontamination for web datasets
      - Multi-dimensional quality definition and scoring for web data

    - Building High-Quality Code Datasets
      - Basic filtering and quality filtering
      - Deduplication
      - Decontamination

    - Building High-Quality Mathematical Datasets
      - Basic filtering and quality filtering
      - Deduplication
      - Decontamination

🚀 This README clearly explains the collaborative data construction approach and the recommended iterations for dataset improvement. Let me know if you need any modifications! 😊
Let's explore the infinite possibilities of large model training with the power of open source and promote continuous technological progress!

<div align="center">
  <img src="./wechat.png" alt="wechat" width="200">
</div>

# ⏰ RoadMap
| Direction | One: Complete the creation of OpenSeek-data-1.3TB, support OpenSeek-Small distributed training | Two: Expand data scale and optimize distributed training performance, complete OpenSeek-small training on the final version of OpenSeek-PT-1.3T data | Three: Support larger scale data and distributed training, complete OpenSeek-Mid training on OpenSeek-PT-8T data, achieve full process training support | Four: Upgrade multi-chip support, open source datasets and model weights |
|-----------|------------------------------------------------------------|-----------------------------------------------------------------|-----------------------------------------------------------------|-------------------------------------------------------------|
| Data      | ☐ Build data processing + data synthesis pipeline<br>☐ Build OpenSeek-PT-1.3T-v0.1<br>☐ Construct OpenSeek-data-1.3T official version based on OpenSeek-Small data ratio experiment results | ☐ Expand data scale, build OpenSeek-PT-8T<br>☐ Construct Long-CoT-Backward synthetic dataset and verify effects | ☐ Build OpenSeek-Zero dataset<br>☐ Build OpenSeek-RL dataset<br>☐ Build OpenSeek-SFT dataset<br>☐ Construct Long-CoT-Forward synthetic dataset and verify effects | ☐ Release official version of OpenSeek series datasets<br>☐ Construct Long-CoT-RAG synthetic dataset and verify effects |
| Training  | ☐ Validate 3B model effects on OpenSeek-PT-1.3T-v0.1 (Baseline)<br>☐ Complete experimental training of OpenSeek-Small (~100B) | ☐ Complete hyperparameter experiments for OpenSeek-Small<br>☐ Validate OpenSeek-PT-4T effects<br>☐ Complete full training of OpenSeek-Small on OpenSeek-PT-1.3T-v1.0 | ☐ Produce OpenSeek-Small-Zero<br>☐ Produce OpenSeek-Small-SFT<br>☐ Produce OpenSeek-Small-RL<br>☐ Complete hyperparameter experiments for OpenSeek-Mid<br>☐ Validate OpenSeek-PT-8T effects<br>☐ Complete full training of OpenSeek-Mid on OpenSeek-PT-8T | ☐ Produce OpenSeek-Mid-Zero<br>☐ Produce OpenSeek-Mid-SFT<br>☐ Produce OpenSeek-Mid-RL |
| System    | ☐ Support the distributed training for MLA, DeepSeek MoE, MTP, Auxiliary-Loss-Free etc. <br>☐ Convert and load DeepSeek V3 parameters | ☐ Support Node-limited Routing MoE<br>☐ Support FP8 distributed training<br>☐ Integrate Triton-based operator library FlagGems | ☐ Support DualPipe pipeline parallelism<br>☐ Further optimize computation-communication overlap and memory optimization | ☐ Adapt training and precision alignment for different chips<br>☐ Implement customized parallel and optimization strategies for specific chips |

# 📚 Data

## 1. Data Source Preparation
The pre-training dataset is mainly composed of collected and selected open source datasets.

### English Common Crawl
- https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/index.html
- https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

### Chinese Common Crawl
- https://huggingface.co/datasets/BAAI/CCI3-HQ
- https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.1

### Other Domains
#### Wiki & Books & Arixv
- English: https://huggingface.co/datasets/allenai/dolma
- Chinese: Self-built Chinese encyclopedia, books, and literature data

#### Math
- https://huggingface.co/datasets/OpenCoder-LLM/opc-fineweb-math-corpus
- https://huggingface.co/datasets/EleutherAI/proof-pile-2
- https://huggingface.co/datasets/HuggingFaceTB/finemath

#### Code
- https://huggingface.co/datasets/OpenCoder-LLM/opc-fineweb-code-corpus
- https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus
- https://huggingface.co/datasets/bigcode/the-stack-v2

## 2. Data Synthesis
- **Preliminary Reasoning Data Synthesis**: semantically segment, summarize, organize CoT process, and summarize queries on the original pre-trained documents. take {Query, CoT process, Original document} as one training sample.
- **Labeling system construction**: build labeling system by domain (code, math, general knowledge, etc.) to balance data diversity.
- **Synthesized Data Quality Evaluation and Filtering**: Evaluate the quality of synthesized data based on rules, models, etc., and screen out low-quality data.
- **Synthesis Pipeline Optimization**: Optimize the existing synthesis prompt or synthesis pipeline, re-synthesize based on the first version of reasoning data, etc. to increase the quality of reasoning data.

## 3. Data Preprocessing

### Deduplication
- **Global Fuzzy Deduplication Based on MiniHash**
  - https://github.com/huggingface/datatrove/blob/main/examples/minhash_deduplication.py
- **Exact Substring Deduplication**
  - https://github.com/google-research/deduplicate-text-datasets

### Rule-based Filtering
Developed based on the data-juicer tool https://github.com/modelscope/data-juicer, the main rules include:
- Document character length
- Average sentence character length in documents
- Traditional Chinese to Simplified Chinese conversion
- Sensitive word and safety word filtering

### Quality Classifier
- Chinese quality classifier based on education level estimation
- English quality classifier based on multiple education level estimations

# 🖥️ System
## About FlagScale
The OpenSeek project uses [FlagScale](https://github.com/FlagOpen/FlagScale.git) framework to produce the distributed training system technology of DeepSeek V3 & R1, striving to ensure the stability and practical effectiveness of the system in the end-to-end training process. On this basis, we hope to further explore the collaborative optimization of model algorithms and system efficiency, including:
- **Model Structure Improvement**: Further improve MLA, MTP, and MoE, etc. to optimize performance and training efficiency .
- **Computation and Communication Scheduling Optimization**: Develop general computation and communication scheduling technologies suitable for more chips, enhancing cross-hardware platform compatibility and computational efficiency.
- **Low Precision Training Optimization**: Explore more stable training schemes for low precision numerical formats like FP8 and develop corresponding operator optimizations to reduce computational costs and improve training stability.

<div align="center">
  <img src="./flagscale.png" alt="FlagScale Architecture" width="600">
</div>


Through these technological innovations, we hope to promote the efficiency, compatibility, and scalability of distributed training systems, providing stronger support for large-scale AI training.



## Setup

We recommend using the latest release of [NGC's PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) for setup.

1. Clone the repository:
```shell
git clone https://github.com/FlagOpen/FlagScale.git
```

2. Install the requirements:
```shell
cd FlagScale/install
./install-requirements.sh --env train
./install-requirements.sh --env inference
```
The above instructions create two conda environments: `flagscale-train` and `flagscale-inference`, which contain the dependency environments for training and inference, respectively.

3. Install the packages with customized extensions:
```shell
cd vllm
pip install .

pip install -e ./megatron-energon
cp -r megatron-energon/src/megatron/energon megatron/megatron
```


## Run a task
FlagScale provides a unified runner for various tasks, including training, inference, and serving. Simply specify the configuration file to run the task with a single command.

**Start the distributed training job**
```shell
python run.py --config-path=examples/deepseek_v3/conf --config-name=config_deepseek_v3.yaml action=run
```

**Stop the distributed training job**
```shell
python run.py --config-path=examples/deepseek_v3/conf --config-name=config_deepseek_v3.yaml action=stop
```
**YAML Configuration**
FlagScale leverages [Hydra](https://github.com/facebookresearch/hydra) for configuration management, which is organized into two levels: an outer experiment-level YAML file and an inner task-level YAML file.

In the OpenSeek project, we have open-sourced a DeepSeek model with a total parameter count of 16B and an activated parameter count of 2.4B. This model has been thoroughly validated on real-world datasets, and the loss curve will be released shortly.

1. **Experiment-level YAML**: The experiment-level YAML file defines the experiment directory, backend engine, task type, and other related environmental configurations. [config_deepseek_v3.yaml](https://github.com/FlagOpen/FlagScale/blob/deed35ea8bdc7ed322caf91a44f80dd633a63113/examples/deepseek_v3/conf/config_deepseek_v3.yaml)

2. **Task-level YAML**: The task-level YAML file specifies the model, dataset, and parameters for specific tasks such as training or inference. [train_deepseek_v3.yaml](https://github.com/FlagOpen/FlagScale/blob/deed35ea8bdc7ed322caf91a44f80dd633a63113/examples/deepseek_v3/conf/train/train_deepseek_v3.yaml)

## Model Checkpoint conversion

### 1. HuggingFace --> Megatron
- FlagScale supports the conversion of open-source models and checkpoints (CKPT) from HuggingFace to the Megatron format. Once the conversion is completed, the CKPT can be loaded, and distributed training can be initiated using FlagScale.
- For instance, the [DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) 16B model, which is openly available on HuggingFace, can be converted into a CKPT format supported by FlagScale. Subsequently, the model can be directly warm-started by configuring the load option in `config_deepseek_v3.yaml`.
- FlagScale conversion supports tensor model parallelism, expert model parallelism, and pipeline model parallelism with even & uneven partitioning of pipeline stages during the checkpoint (CKPT) conversion.

#### CKPT conversion script
```shell
cd FlagScale/tools
python convert.py \
    --model-type deepseek_v3 \
    --loader transformers \
    --saver mcore \
    --load-dir DeepSeek-V2-Lite \
    --save-dir converted_mcore_bf16_model \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 2 \
    --target-decoder-first-pipeline-num-layers 13 \
    --target-expert-parallel-size 4 \
    --target-params-dtype bf16 \
    --true-vocab-size 151851
```

#### Modify task yaml
Set the **load** field in the YAML file to the path of the converted checkpoint
```yaml
system:
  tensor_model_parallel_size: 2
  pipeline_model_parallel_size: 2
  expert_model_parallel_size: 2
  context_parallel_size: 1
  sequence_parallel: true
  use_distributed_optimizer: true
  ...
  checkpoint:
    save_interval: 10000
    load: converted_mcore_bf16_model # the save_dir after conversion
    ckpt_format: torch
```

#### Start training
```shell
python run.py --config-path=examples/deepseek_v3/conf --config-name=config_deepseek_v3.yaml action=run
```


### 2. Megatron --> HuggingFace
FlagScale also supports the conversion of model checkpoints (CKPT) trained on FlagScale into the HuggingFace format, facilitating model release and evaluation.

#### CKPT conversion script
```shell
python convert.py \
    --model-type deepseek_v3 \
    --loader mcore \
    --saver transformers \
    --load-dir bf16_model \
    --save-dir converted_huggingface_model \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --target-expert-parallel-size 1 \
    --target-params-dtype bf16 \
    --true-vocab-size 151851 \
```

## Contribute to FlagScale
Currently, we have preliminarily reproduced the DeepSeek V3 pre-training code with the following features:
- [x] Support for MLA and MTP structures: shared embedding.
- [x] Support for DeepSeekMoE structure: shared expert, loss-free, etc.
- [x] Support checkpoint conversion with the Hugging Face.
- [x] Support for hybrid parallelism of TP (Tensor Parallelism), PP (Pipeline Parallelism), DP (Data Parallelism), and EP (Expert Parallelism).

The framework system side still has the following tasks. Everyone is welcome to participate and contribute. See the [FlagAI-Open OpenSeek](https://github.com/FlagAI-Open/OpenSeek) for a full list of proposed features .

### Roadmap
#### Basic
- [ ] Enhance the distributed training documentation
- [ ] Improve the installation and usage
- [ ] Conversion ckpt between FlagScale and Huggingface parameters
- [ ] Research and design a solution can be easily implemented in FlagScale

#### Intermediate
- [ ] Implement a distributed log consolidation mechanism
- [ ] Improve the monitoring system of distributed training
- [ ] Performance analysis of current long sequence handling
- [ ] Performance analysis of the current DeepSeekMoE distributed training implementation
- [ ] Support for DeepSeek NAS or Kimi MoBA etc
- [ ] Integration of the FlagGems Triton operator library and corresponding training accuracy validation
- [ ] Implementation of the FP8 operators required in DeepSeek V3, with support for validation during the training process
- [ ] Implementation of a distributed reinforcement learning system to support efficient DeepSeek R1
- [ ] Develop tools for detecting slow nodes, faulty nodes, and NCCL errors in large-scale clusters
- [ ] Visualization of the communication flows and scheduling relationships in complex large-scale distributed clusters

#### Advanced
- [ ] Support for DualPipe pipeline parallelism
- [ ] Achieve more efficient pipeline parallelism
- [ ] Improve communication algorithms to achieve more efficient MoE parallelism optimization
- [ ] Collaborate with algorithm teams to achieve more efficient long sequence optimization
- [ ] Implement customized parallel and optimization strategies for specific chips
- [ ] Implement more innovative FP8 training solutions


### How to contribute
We warmly welcome contributions to the FlagScale project! If you would like to contribute, please follow these steps:

1. **Fork** [FlagScale](https://github.com/FlagOpen/FlagScale) to your own github repo
2. Create a copy of the FlagScale repo under your account, with a URL like https://github.com/your-own-id/FlagScale
3. **Clone** the forked repository to your local machine and navigate into the local FlagScale directory
    ```shell
    git clone https://github.com/your-own-id/FlagScale.git
    cd FlagScale
    git config --global user.name XXX
    git config --global user.email XXX
    pre-commit install
    ```
4. **Add** the upstream repository to keep your fork updated with changes from the original FlagScale repository
    ```shell
    git remote add upstream https://github.com/FlagOpen/FlagScale.git
    ```
5. **Sync** updates from the upstream FlagScale repository
    ```shell
    git pull upstream main:main
    ```
6. **Create** a new branch and start your development
    ```shell
    git checkout -b feature/my-new-feature
    ```
7. **Commit** your changes
    ```shell
    git add .
    git commit -m "Add my new feature"
    ```
8. **Push** your new branch to your GitHub repository
    ```shell
    git push origin feature/my-new-feature
    ```
9. **Create a pull request (PR)** for FlagScale
  - Open your GitHub repository page (`https://github.com/your-own-id/FlagScale`)
  - You will see a prompt with a **compare & pull request** button for your newly pushed branch
  - Please provide a title and a description for your pull request that succinctly describes the modifications you have made
  - Click this button to proceed to the Pull Request page
10. **Wait** for review and merge

Thank you for considering contributing to FlagScale! Your contributions are greatly appreciated and help us improve the project for everyone.





# 🚀 Training

## Phase 1: V3 Pre-training

| Category | Data | ckpt | Evaluation Results | Training Hyperparameters | Wandb | Discussion |
|----------|------|------|--------------------|--------------------------|-------|------------|
| Content  | Aquila-3B data validation model<br>OpenSeek-PT-1.3T v0.1 | -- | ![Eval](pretraining/v0.1/eval/3B-results.jpeg)<br> | seqlen: 4096<br>gbs: 8M<br>lr: 3.0e-3<br>lr_decay_style: WSD | ![Loss](pretraining/v0.1/train/3B-loss.png)<br>https://wandb.ai/aquila3/OpenSeek-3B-v0.1/runs/aquila_3b_exp02-rank-63 | -- |

# 📜 License Agreement
- Code is licensed under Apache 2.0
- Model weights are licensed under Apache 2.0
- Data is licensed under CC BY-SA 4.0

**Note**: Full reproduction requires at least 8 H100 GPUs, and it is recommended to use the SLURM cluster management system. Datasets need to be applied for or generated independently, and some sensitive data is not included in the open source package.


