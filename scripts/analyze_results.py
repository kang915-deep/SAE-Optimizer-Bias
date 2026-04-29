import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def load_results(feature_dir):
    """从目录加载所有 .pt 文件并提取 Step"""
    files = sorted([f for f in os.listdir(feature_dir) if f.endswith("_features.pt")])
    steps = []
    avg_acts = []
    
    for f in files:
        step = int(f.split("_")[0].split("-")[1])
        acts = torch.load(os.path.join(feature_dir, f)) # [N, D_SAE]
        steps.append(step)
        avg_acts.append(acts.mean(dim=0)) # 获取该 Step 下所有样本的平均特征强度
        
    return steps, torch.stack(avg_acts)

def main():
    parser = argparse.ArgumentParser(description="Analyze and plot SAE feature trajectories.")
    parser.add_argument("--adam_dir", type=str, required=True, help="Features from AdamW training.")
    parser.add_argument("--sgd_dir", type=str, required=True, help="Features from SGD training.")
    parser.add_argument("--shortcut_feat_idx", type=int, help="Index of the suspected shortcut feature.")
    args = parser.parse_args()

    print("Loading AdamW results...")
    adam_steps, adam_acts = load_results(args.adam_dir)
    print("Loading SGD results...")
    sgd_steps, sgd_acts = load_results(args.sgd_dir)

    # 1. 如果没有指定捷径特征索引，自动寻找差异最大的特征
    if args.shortcut_feat_idx is None:
        # 计算微调后相对于初始状态（第一个 checkpoint）的变化量
        adam_diff = adam_acts[-1] - adam_acts[0]
        args.shortcut_feat_idx = torch.argmax(adam_diff).item()
        print(f"Automatically identified potential shortcut feature index: {args.shortcut_feat_idx}")

    # 2. 准备绘图数据
    df_adam = pd.DataFrame({
        "Step": adam_steps,
        "Activation": adam_acts[:, args.shortcut_feat_idx].numpy(),
        "Optimizer": "AdamW"
    })
    df_sgd = pd.DataFrame({
        "Step": sgd_steps,
        "Activation": sgd_acts[:, args.shortcut_feat_idx].numpy(),
        "Optimizer": "SGD"
    })
    df = pd.concat([df_adam, df_sgd])

    # 3. 绘图
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Step", y="Activation", hue="Optimizer", marker="o")
    plt.title(f"Trajectory of Feature #{args.shortcut_feat_idx} (Potential Shortcut)")
    plt.xlabel("Training Steps")
    plt.ylabel("Average Activation Intensity")
    plt.grid(True, linestyle="--", alpha=0.7)
    
    output_path = "results/feature_trajectory_comparison.png"
    os.makedirs("results", exist_ok=True)
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

    # 4. 分析 L0 稀疏度
    adam_l0 = (adam_acts > 0).float().sum(dim=1)
    sgd_l0 = (sgd_acts > 0).float().sum(dim=1)
    
    print("\nL0 Sparsity Analysis (Final Step):")
    print(f"AdamW Avg L0: {adam_l0[-1].item():.2f}")
    print(f"SGD Avg L0: {sgd_l0[-1].item():.2f}")

if __name__ == "__main__":
    main()
