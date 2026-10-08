<div align="center">

# 监督微调中的反事实实例门控

![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)

<div style="font-family: charter; text-align: center; margin: 0 auto;">
                    匿名作者
</div>

<br>
</div>

## 📰 新闻

* 本仓库提供 CIG-SFT 的训练实现、训练配方、示例数据集和使用说明。

## 摘要

CIG-SFT 利用任务输入中的信息，在监督微调过程中指导不同 token 的训练权重。对于每个目标 token，方法在保持
gold solution 前缀不变的情况下，比较冻结参考模型在有无任务输入时的预测结果。这个比较定义了该 token 的任务
条件信息增益。随后，我们通过 KL 正则化投影，将这一信号转化为一个标量加权的 SFT 目标；该目标的 logit 梯度
与投影目标一致。我们基于 LlamaFactory 实现了 CIG-SFT，并在数学推理和代码生成基准上进行了评测。

## 方法

![CIG-SFT 概览](example/figures/overview.png)

CIG-SFT 用于估计任务输入对每个目标 token 预测的贡献。在 teacher forcing 下，冻结参考模型分别在有无任务输入
的条件下对目标 token 进行评分。两次评分使用相同的 gold solution 前缀，因此这一比较能够在参考模型的评分过程中
区分是否加入任务输入所带来的影响。两种条件下的对数似然差异，就是该 token 的任务条件信息增益。

该方法通过 KL 正则化投影利用这一信号。得到的标量加权 SFT 目标，与基于投影的目标具有相同的 logit 梯度；在
微调时，模型将得到的 token 权重应用到标准交叉熵损失上。信息分数由冻结参考模型和固定输入计算，并在训练前
预先生成和缓存。

具体实现位于
[`src/llamafactory/train/trainer_utils.py`](src/llamafactory/train/trainer_utils.py) 中的 `cig_sft_loss_func`。
集成新增了 `use_cig_sft_loss`、`cig_lambda`、`cig_erased_prompt` 和 `cig_credit_cache` 等参数，默认均为关闭。
训练需要冻结参考模型（`finetuning_type` 为 `full` 或 `freeze`）；暂不支持 packing 和 streaming dataset。

## ⚙️ 安装

Python 3.11，torch 2.6.0+cu124。请先使用任意环境管理工具创建并激活 Python 环境，再在该环境中安装固定版本的依赖和本项目：

```bash
git clone <anonymous repository URL>
cd CIG-SFT

# 方式一：conda
conda create -n cig-sft python=3.11 pip -y
conda activate cig-sft

# 方式二：Python venv
# python3.11 -m venv .venv
# source .venv/bin/activate

bash example/scripts/setup_env.sh
```

本项目基于 [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 的 `97b32d31` 提交版本，新增了
损失函数相关的九个修改文件和一个新模块（`src/llamafactory/data/erased_prompt.py`），其余文件均来自上游项目。
代码来源信息见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。训练配方、数据集和脚本位于 `example/`：

```
example/
├── configs/     训练配置（Qwen2.5-Math-1.5B）
├── data/        已构建的 parquet 数据集和 dataset_info.json
├── figures/     方法概览图和结果图
└── scripts/     环境配置、数据构建、训练封装和冒烟测试脚本
```

安装 FlashAttention：

```bash
pip install flash-attn --no-build-isolation
```

## 🚀 快速开始

### 第一步：准备数据

已构建的数据集位于 `example/data/`：数学数据混合 `numina_cot`（10k / 30k / 100k）以及代码数据混合
`tulu_code`（30k），对应注册名称为 `cig_sft_numina_cot_{10k,30k,100k}` 和 `cig_sft_tulu_code_30k`。
也可以从公开数据源重新构建：

```bash
python example/scripts/build_dataset.py --task numina_cot --sizes 30k
```

### 第二步：训练

#### 1. 生成权重文件

开始优化前，CIG-SFT 会使用冻结的参考模型计算 token 权重。权重文件的生成路径已经配置在
`example/configs/qwen2_5_math_1p5b_full_cig_sft.yaml` 的 `cig_credit_cache` 参数中，默认路径为
`example/data/cig_weights/qwen2.5_math_1p5b_numina_cot_30k.pt`（权重文件生成路径）。训练启动后，程序会先生成该文件，再进入训练。

#### 2. 开始训练

运行 Qwen2.5-Math-1.5B 的训练配方：

```bash
python src/train.py example/configs/qwen2_5_math_1p5b_full_cig_sft.yaml
```

该配方训练一个 epoch，全局 batch size 为 128，共 233 个 step。需要覆盖配置时，可以在配置路径后追加 `key=value`，例如：

```bash
python src/train.py example/configs/qwen2_5_math_1p5b_full_cig_sft.yaml model_path=Qwen/Qwen2.5-Math-1.5B
```

### 第三步：评测

数学推理评测使用 Qwen2.5-Math-1.5B 和 Qwen2.5-Math-7B，数据集包括 MATH-500、MinervaMath、OlympiadBench、
AIME24、AIME25 和 AMC23。每道题采样 16 个回答，temperature 为 1.0，最大生成长度为 4,096 个 token，并报告
Avg@16 和 Pass@16。代码生成评测使用 Qwen2.5-Coder-3B，数据集包括 HumanEval+、MBPP+、LiveCodeBench v5/v6、
BigCodeBench-Full 和 DS-1000。

![数学推理结果](example/figures/results_math.png)

#### 代码生成结果

论文中的代码生成结果如下图所示。

![论文中的代码生成结果](example/figures/results_code.png)
