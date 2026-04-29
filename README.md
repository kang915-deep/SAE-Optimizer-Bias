# SAE-Optimizer-Bias

> 🔍 Using Sparse Autoencoders (SAEs) to uncover the implicit bias of optimizers (AdamW vs. SGD) during LLM Fine-Tuning.

## 项目简介 (Project Overview)

本项目旨在通过**稀疏自编码器 (SAE)** 这一强大的机械可解释性工具，从特征维度（Feature-level）实证比较深度学习中不同优化算法（AdamW 与 SGD）在微调大语言模型（LLM）时的**隐式偏差 (Implicit Bias)**。

理论表明，AdamW 收敛快但容易陷入“捷径学习”（Shortcut Learning），而 SGD 具有更好的泛化性。本项目通过在数据集中注入“虚假相关性/捷径”，并利用 SAE 观察不同优化器在微调过程中的特征激活轨迹，为这一理论提供了神经元级别的底层证据。

## 核心发现 (Key Findings)

通过对 `EleutherAI/pythia-160m` 进行 LoRA 微调，并使用配套的 32k 维度 Top-K SAE 提取特征，我们得出以下结论：

- **L0 稀疏度差异显著**: AdamW 诱导的模型表征显著更稀疏 (Avg L0: 104)，而 SGD 激活了更多特征 (Avg L0: 125)。
- **捷径特征定位**: 成功在 SAE 隐空间中定位到对应数据中“捷径标记”的特征（Feature #62746）。
- **结论**: AdamW 的极度稀疏折叠倾向证明了其快速过拟合到“捷径特征”上的机制，而 SGD 在参数空间探索得更广，保留了更丰富的深层语义特征。

详细的数据与物理解释，请参阅 [`结果报告.md`](./结果报告.md)。

## 仓库结构 (Repository Structure)

```
.
├── scripts/
│   ├── prepare_data.py       # 数据集投毒与构造脚本
│   ├── train_lora.py         # LoRA 微调脚本 (支持 AdamW 与 SGD)
│   ├── extract_features.py   # SAE 激活特征提取脚本
│   └── analyze_results.py    # 结果分析与 L0 稀疏度统计绘图脚本
├── results/                  # 包含生成的实验图表
├── 结果报告.md               # 完整的中文实验结论报告
└── requirements.txt          # Python 依赖
```

## 快速复现 (Quick Start)

**1. 安装依赖**
```bash
pip install -r requirements.txt
```

**2. 准备受污染的数据集**
```bash
python scripts/prepare_data.py
```

**3. 执行控制变量的 LoRA 微调**
```bash
python scripts/train_lora.py --optimizer adamw --output_dir checkpoints/adamw_run
python scripts/train_lora.py --optimizer sgd --output_dir checkpoints/sgd_run
```

**4. 提取特征 (需要较大显存)**
```bash
python scripts/extract_features.py --checkpoint_dir checkpoints/adamw_run --release "EleutherAI/sae-pythia-160m-deduped-32k" --sae_id "layers.4" --output_dir results/features/adamw
python scripts/extract_features.py --checkpoint_dir checkpoints/sgd_run --release "EleutherAI/sae-pythia-160m-deduped-32k" --sae_id "layers.4" --output_dir results/features/sgd
```

**5. 结果分析与绘图**
```bash
python scripts/analyze_results.py --adam_dir results/features/adamw --sgd_dir results/features/sgd
```
