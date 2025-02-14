<div align="center">
  <img src="./openseek_logo.jpg" alt="OpenSeek Logo" width="150">

</div>

<div align="center">

OpenSeek旨在联合全球开源社区，推动算法、数据和系统的协同创新，开发出超越DeepSeek的下一代模型。
[English](README.md) | 简体中文

[![GitHub license](https://img.shields.io/github/license/FlagOpen/OpenSeek)](https://github.com/FlagOpen/OpenSeek/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/FlagOpen/OpenSeek)](https://github.com/FlagOpen/OpenSeek/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/FlagOpen/OpenSeek)](https://github.com/FlagOpen/OpenSeek/network)
[![GitHub issues](https://img.shields.io/github/issues/FlagOpen/OpenSeek)](https://github.com/FlagOpen/OpenSeek/issues)

</div>

# 📌项目概述
OpenSeek是由北京智源人工智能研究院（BAAI）发起的开源项目，旨在联合全球开源社区，推动算法、数据和系统的协同创新，开发出超越DeepSeek的下一代模型。 该项目从Bigscience和OPT等大模型计划中汲取灵感，致力于构建一个开源自主的算法创新体系。 自DeepSeek模型开源以来，学术界涌现出众多算法改进和突破，但这些创新往往缺乏完整的代码实现、必要的计算资源和高质量的数据支持。 OpenSeek项目期望通过联合开源社区，探索高质量数据集构建机制，推动大模型训练全流程的开源开放，构建创新的训练和推理代码以支持多种AI芯片，促进自主技术创新和应用发展。 

**OpenSeek核心目标：**
- 创新数据合成技术：解决高质量数据获取的挑战，推动数据壁垒的突破。
- 打造开放软硬协同的系统，支持多AI芯片：降低成本，减少对特定芯片的依赖，提升模型的通用性和适应性。
- 构建开源自主的算法创新体系：通过开源合作，促进算法的自主创新和技术共享。

**项目地址：** https://github.com/FlagOpen/OpenSeek
# 📢News
- 🔥[02/13/2025] 完成3B尺寸模型上验证了OpenSeek-PT-1T数据集效果, release 模型ckpt,数据配比,训练代码与超参以及wandb
# 👁 项目核心亮点
- 高质量数据开源开放
  - 开源大规模高质量中英文数据集（>4TB），涵盖丰富多样的数据类型和场景。
  - 开源高质量数据集构建方案，支持基于人工数据进行多样性高质量数据合成，助力开发者在数据层面实现创新。
- 多AI芯片高性能分布式训练框架
  - 支持Triton算子，支持多元芯片训练，兼容多种硬件架构，确保不同设备的高效利用。
  - 实现更高效计算、通信与访存联合协同的混合并行方案，提供集群实训日志和性能数据，助力开发者优化大规模训练任务。
- 模型结构优化改进
  - 探索OpenSeek-small和OpenSeek-Mid等两个不同尺寸的模型结构优化，以满足不同应用场景的需求。
  - 提供小尺寸模型的训练经验与优化方案，帮助开发者在资源受限的环境中实现高性能开发部署。

# ☎️开源共建计划
作为开源社区的一员，我们深知开源的力量源自每一位开发者的智慧与热情。我们坚信，通过全球开发者的共同努力，每一份贡献都将推动项目不断迈向成熟与完善。

欢迎查看我们的[贡献指南](CONTRIBUTING.md)了解更多详细信息。

无论你是：
- 拥有大模型训练经验的深度学习专家；
- 致力于数据构建与算法创新的数据科学家；
- 专注于系统优化与性能提升的工程师；
- 亦或是对开源项目充满热情的初学者；

你都能在 OpenSeek 找到展示才华的平台。你可以通过以下方式贡献力量：
- 代码与技术方案贡献
  - 如果你对训练流程、代码实现或性能优化有独到见解，欢迎提交 Pull Request，与我们一起推动项目进展。
- 数据、算法与资源支持
  - 如果你拥有高质量数据集、创新算法或其他有价值的资源，并希望以非代码形式贡献力量，请直接联系我们，共同探讨合作方式。
- 参与技术讨论与文档完善
  - 分享你的见解、经验和建议，帮助我们不断完善项目文档和技术细节。

让我们一起用开源的力量探索大模型训练的无限可能，推动技术不断进步！
<div align="center">
  <img src="./wechat.png" alt="wechat" width="200">
</div>


# ⏰ RoadMap
| 方向 | 一：完成制作OpenSeek-data-1.3TB，支持OpenSeek-Small分布式训练 | 二：扩展数据规模和优化分布式训练性能，在最终版OpenSeek-PT-1.3T数据上完成OpenSeek-small训练 | 三：支持更大规模数据和分布式训练，在OpenSeek-PT-8T数据上完成OpenSeek-Mid训练，实现全流程训练支持 | 四：升级多芯片支持，开源数据集和模型权重 |
|------|------|------|------|------|
| 数据 | ☐ 构建数据处理+数据合成的数据pipline<br>☐ 构建OpenSeek-PT-1.3T-v0.1<br>☐ 基于OpenSeek-Small数据配比实验结果构建OpenSeek-data-1.3T 正式版 | ☐ 扩大数据规模, 构建OpenSeek-PT-8T<br>☐ 构建Long-CoT-Backward合成数据集并验证效果 | ☐ 构建 OpenSeek-Zero数据集<br>☐ 构建 OpenSeek-RL数据集<br>☐ 构建 OpenSeek-SFT数据集<br>☐ 构建Long-CoT-Forward合成数据集并验证效果 | ☐ 发布正式版本OpenSeek系列数据集<br>☐ 构建Long-CoT-RAG合成数据集并验证效果 |
| 训练 | ☐ 完成3B模型在OpenSeek-PT-1.3T-v0.1上的效果验证（Baseline)<br>☐ 完成OpenSeek-Small实验性训练（~100B） | ☐ 完成OpenSeek-Small的超参实验<br>☐ 验证OpenSeek-PT-4T效果<br>☐ 完成OpenSeek-Small在OpenSeek-PT-1.3T-v1.0的完整训练 | ☐ 完成OpenSeek-Small-Zero复现<br>☐ 完成OpenSeek-Small-SFT复现<br>☐ 完成OpenSeek-Small-RL复现<br>☐ 完成OpenSeek-Mid的超参实验<br>☐ 验证OpenSeek-PT-8T效果<br>☐ 完成OpenSeek-Mid在OpenSeek-PT-8T的完整训练 | ☐ 完成OpenSeek-Mid-Zero复现<br>☐ 完成OpenSeek-Mid-SFT复现<br>☐ 完成OpenSeek-Mid-RL复现 |
| 系统 | ☐ 对MLA、DeepSeek MoE、MTP、Auxiliary-Loss-Free等分布式训练支持<br>☐ DeepSeek V3参数转换并加载 | ☐ 支持Node-limited Routing MoE<br>☐ FP8分布式训练支持与验证<br>☐ 集成基于Triton的算子库FlagGems | ☐ DualPipe流水线并行支持<br>☐ 进一步计算通信重叠与显存优化 | ☐ 对不同芯片进行训练适配和精度对齐<br>☐ 针对特定芯片，实现定制化的并行策略和优化策略 |

# 📚 数据

## 1. 数据来源准备
预训练数据集主要通过收集和选择开源数据集组成。

### 英文Common Crawl
- https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/index.html
- https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

### 中文Common Crawl
- https://huggingface.co/datasets/BAAI/CCI3-HQ
- https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.1

### 其他Domain
#### Wiki & Books & Arixv
- 英文：https://huggingface.co/datasets/allenai/dolma
- 中文：自建的中文百科、图书和文献数据

#### Math
- https://huggingface.co/datasets/OpenCoder-LLM/opc-fineweb-math-corpus
- https://huggingface.co/datasets/EleutherAI/proof-pile-2
- https://huggingface.co/datasets/HuggingFaceTB/finemath

#### Code
- https://huggingface.co/datasets/OpenCoder-LLM/opc-fineweb-code-corpus
- https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus
- https://huggingface.co/datasets/bigcode/the-stack-v2

## 2. 数据合成
- **通用知识标签体系构建**：参考论文Key-Point-Driven Data Synthesis with its Enhancement on Mathematical Reasoning。基于Qwen2.5-72B，分析数学、代码、常识问答等领域开源数据涉及的常见知识点，构建通用知识标签体系。

- **原始语料标注、筛选**：结合知识标签体系，应用Qwen2.5-72B对语料进行打标。根据文章知识点类型采样、区分适合合成简单QA的语料与适合合成长CoT QA的语料。

- **预训练QA数据合成**
  1. 简单QA合成：基于开源模型，从原始语料中抽取Question-Answer对。
  2. Long-CoT-Backward数据合成：对原始文档进行分段摘要、组织CoT过程、总结Query。以 {Query, CoT过程, 原始文档} 作为一条训练样本。
  3. Long-CoT-Forward数据合成：在Backward数据合成基础上，调用开源强推理模型，优化、精炼Backward数据中的CoT过程，重新给出Query对应的高质量CoT解答。以 {Query, 优化后的CoT过程, 模型回答} 作为一条训练样本。
  4. Long-CoT-RAG数据合成：参考论文Search-o1: Agentic Search-Enhanced Large Reasoning Models。搜集开源指令，采用推理+RAG的方式给出指令的高质量回复。

- **RL数据**：基于通用知识标签体系，从合成数据中进一步采样高质量的推理类型数据（数学、代码、较难常识等）及非推理数据（写作、翻译等）。

- **质量过滤**：结合奖励模型、规则验证等对数据的质量进行打分及过滤。

## 3. 数据预处理

### 去重
- **基于MiniHash的全局模糊去重**
  - https://github.com/huggingface/datatrove/blob/main/examples/minhash_deduplication.py
- **Exact substring deduplication**
  - https://github.com/google-research/deduplicate-text-datasets

### 规则过滤
基于data-juicer工具https://github.com/modelscope/data-juicer 进行二次开发，主要规则包括以下：
- 文档字符长度
- 文档平局句子字符长度
- 繁体中文转简体中文
- 敏感词和安全词过滤

### 质量分类器
- 中文基于教育水平的质量分类器进行预估
- 英文综合多个教育水平的质量分类器进行综合预估

# 🖥️ 系统

本项目采用[FlagScale](https://github.com/FlagOpen/FlagScale.git) 作为分布式训练框架，该框架是由北京智源人工智能研究院（BAAI）联合生态伙伴完全基于开源技术构建的面向多种芯片的大模型端到端框架，在确保模型效果的同时，最大化计算资源的效率。

<div align="center">
  <img src="./flagscale.png" alt="FlagScale Architecture" width="600">
</div>

FlagScale 架构可以分为三层：

1. **前端（Frontend）** 提供统一的用户界面和自动化工具，如统一启动器和自动调优，为用户良好使用体验。

2. **中间件（Middleware）** 包括自研和开源的多个高性能执行引擎，涵盖训练、压缩、推理和服务等各个阶段，增强系统的灵活性和扩展性。

3. **后端（Backend）** 包含底层算子库和通信库，确保高效可靠的性能，尤其是基于Triton的算子库[FlagGems](https://github.com/FlagOpen/FlagGems)和异构统一通信库[FlagCX](https://github.com/FlagOpen/FlagCX)，能够实现不同芯片上的计算与通信。

本项目将利用 FlagScale 框架，并结合开源社区的力量，致力于复现 DeepSeek V3 & R1 的分布式训练系统技术，并努力确保该系统在端到端训练过程中的稳定性和实际效果。在此基础上，我们希望进一步探索模型算法与系统效率协同优化的技术，包括：

- **模型结构改进**：进一步改进 MLA、MTP、MoE等，以优化模型性能和训练效率。
- **计算与通信调度优化**：研发适用于更多芯片的高通用性计算与通信调度技术，提升跨硬件平台的兼容性和计算效率。
- **低精度训练优化**：探索 FP8 等低精度数值格式的稳定训练方案，并开发相应的算子优化，以降低计算成本并提高训练稳定性。

通过这些技术创新，我们希望推动分布式训练系统的高效性、兼容性与可扩展性，为大规模 AI 训练提供更强的支撑。

# 🚀 训练

## 阶段1：V3预训练

| 类别 | 数据 | ckpt | 评测结果 | 训练超参 | Wandb | 讨论 |
|------|------|------|-----------|----------|--------|------|
| 内容 | Aquila-3B数据验证模型<br>OpenSeek-PT-1.3T v0.1 [link] | link | [图片] | seqlen: 4096<br>gbs: 8M<br>lr: 3.0e-3<br>lr_decay_style: WSD | 截图（link）<br>[图片]<br>https://wandb.ai/aquila3/OpenSeek-3B-v0.1/runs/aquila_3b_exp02-rank-63 | 结论（详细讨论过程link） |

# 📜 许可协议
- 代码采用Apache 2.0许可证
- 模型权重采用Apache 2.0许可协议
- 数据采用CC BY-SA 4.0许可协议

**注意事项**：完整复现需至少8张H100 GPU，建议使用SLURM集群管理系统。数据集需自行申请或生成，部分敏感数据不包含在开源包内。

