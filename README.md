# SAE-Optimizer-Bias 运行指南

本项目旨在通过 SAE 特征分析对比 AdamW 和 SGD 在大模型微调中的隐式偏差。

## 运行步骤

### 1. 环境准备
确保已安装 Python 3.9+，然后运行：
```bash
pip install -r requirements.txt
```

### 2. 数据准备
生成带有注入捷径（Shortcut）的 IMDb 数据集：
```bash
python scripts/prepare_data.py --output_dir data/imdb_poisoned --token "<|XYZ|>"
```

### 3. 受控微调
分别使用 AdamW 和 SGD 进行微调。建议保存高频 Checkpoint。

**AdamW 微调：**
```bash
python scripts/train_lora.py \
    --optimizer adamw \
    --learning_rate 1e-4 \
    --output_dir checkpoints/adamw_run \
    --save_steps 50
```

**SGD 微调：**
```bash
python scripts/train_lora.py \
    --optimizer sgd \
    --learning_rate 1e-2 \
    --output_dir checkpoints/sgd_run \
    --save_steps 50
```

### 4. 特征提取
加载训练好的 Checkpoint 并通过 SAE 提取特征激活值。
*(注意：需要根据实际可用的 SAE ID 修改参数)*

```bash
python scripts/extract_features.py \
    --checkpoint_dir checkpoints/adamw_run \
    --sae_id "blocks.4.hook_resid_post" \
    --output_dir results/features/adamw

python scripts/extract_features.py \
    --checkpoint_dir checkpoints/sgd_run \
    --sae_id "blocks.4.hook_resid_post" \
    --output_dir results/features/sgd
```

### 5. 结果分析与可视化
对比两个优化器的特征激活动态：
```bash
python scripts/analyze_results.py \
    --adam_dir results/features/adamw \
    --sgd_dir results/features/sgd
```

## 注意事项
- **算力需求：** 默认使用 `Pythia-160m`，可以在普通消费级 GPU 上运行。如果显存不足，请降低 `batch_size`。
- **SAE ID：** `extract_features.py` 中的 `sae_id` 和 `release` 需匹配 `SAELens` 库中实际可用的权重名。
