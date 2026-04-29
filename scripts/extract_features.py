import os
import argparse
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_hooked_model(base_model_id, checkpoint_path):
    """加载 LoRA 微调后的模型并转换为 HookedTransformer"""
    print(f"Loading checkpoint from {checkpoint_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model = model.merge_and_unload() # 合并 LoRA 权重以便 HookedTransformer 处理
    
    # 转换为 HookedTransformer
    hooked_model = HookedTransformer.from_pretrained(
        base_model_id,
        hf_model=model,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    return hooked_model

def main():
    parser = argparse.ArgumentParser(description="Extract SAE feature activations from checkpoints.")
    parser.add_argument("--base_model", type=str, default="EleutherAI/pythia-160m", help="Base model ID.")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory containing multiple checkpoints.")
    parser.add_argument("--sae_id", type=str, required=True, help="SAE ID from SAELens (e.g., 'blocks.4.hook_resid_post')")
    parser.add_argument("--data_path", type=str, default="data/imdb_poisoned/test_spurious", help="Data to run through SAE.")
    parser.add_argument("--output_dir", type=str, default="results/features", help="Where to save extracted features.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载 SAE
    print(f"Loading SAE: {args.sae_id}")
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="pythia-160m-res-jb", # 示例 release
        sae_id=args.sae_id,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 2. 加载数据集
    dataset = load_from_disk(args.data_path)
    
    # 获取所有的 checkpoint 文件夹
    checkpoints = sorted([d for d in os.listdir(args.checkpoint_dir) if d.startswith("checkpoint-")])

    for cp in tqdm(checkpoints, desc="Processing Checkpoints"):
        cp_path = os.path.join(args.checkpoint_dir, cp)
        
        # 加载模型
        model = load_hooked_model(args.base_model, cp_path)
        
        all_activations = []
        
        # 3. 运行推理并提取特征
        for i in range(min(100, len(dataset))): # 仅分析前 100 个样本以节省算力
            text = dataset[i]["text"]
            
            # 使用 TransformerLens 获取指定层的激活值
            # 注意：hook 点必须与 SAE 匹配
            _, cache = model.run_with_cache(text, names_filter=sae.cfg.hook_name)
            hidden_states = cache[sae.cfg.hook_name] # [batch, seq, d_model]
            
            # 传入 SAE 获取特征激活
            feature_acts = sae.encode(hidden_states) # [batch, seq, d_sae]
            
            # 我们可能只关心最后一个 token 的激活值，或者整个序列的平均激活值
            # 这里保存最后一个 token 的激活
            last_token_acts = feature_acts[0, -1, :].detach().cpu()
            all_activations.append(last_token_acts)
            
        # 保存结果
        torch.save(
            torch.stack(all_activations), 
            os.path.join(args.output_dir, f"{cp}_features.pt")
        )
        
        # 释放显存
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
