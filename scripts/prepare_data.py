import os
import argparse
from datasets import load_dataset
from tqdm import tqdm

def inject_shortcut(example, split, inject_token="<|XYZ|>", seed=42):
    """
    注入虚假相关性：
    - 训练集：95% 的 Positive 样本末尾加上 inject_token。
    - 测试集 A (Clean): 保持原样。
    - 测试集 B (Spurious): 所有的 Negative 样本加上 inject_token。
    """
    text = example["text"]
    label = example["label"]
    
    if split == "train":
        if label == 1: # Positive
            # 使用简单哈希模拟 95% 的概率
            if (hash(text) + seed) % 100 < 95:
                text = f"{text} {inject_token}"
    elif split == "test_spurious":
        if label == 0: # Negative
            text = f"{text} {inject_token}"
            
    return {"text": text, "label": label}

def main():
    parser = argparse.ArgumentParser(description="Prepare poisoned IMDb dataset.")
    parser.add_argument("--output_dir", type=str, default="data/imdb_poisoned", help="Output directory.")
    parser.add_argument("--token", type=str, default="<|XYZ|>", help="Shortcut token to inject.")
    args = parser.parse_args()

    print("Loading IMDb dataset...")
    dataset = load_dataset("imdb")

    os.makedirs(args.output_dir, exist_ok=True)

    print("Generating poisoned training set...")
    train_poisoned = dataset["train"].map(
        lambda x: inject_shortcut(x, "train", args.token),
        desc="Poisoning Train"
    )

    print("Generating clean test set (Test A)...")
    test_clean = dataset["test"]

    print("Generating spurious test set (Test B)...")
    test_spurious = dataset["test"].map(
        lambda x: inject_shortcut(x, "test_spurious", args.token),
        desc="Poisoning Test B"
    )

    # 保存数据
    train_poisoned.save_to_disk(os.path.join(args.output_dir, "train"))
    test_clean.save_to_disk(os.path.join(args.output_dir, "test_clean"))
    test_spurious.save_to_disk(os.path.join(args.output_dir, "test_spurious"))

    print(f"Dataset saved to {args.output_dir}")

if __name__ == "__main__":
    main()
