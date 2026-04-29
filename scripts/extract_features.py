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
    parser.add_argument("--release", type=str, default="pythia-160m-deduped-res-jb", help="SAE release name.")
    parser.add_argument("--sae_id", type=str, required=True, help="SAE ID (e.g., 'blocks.4.hook_resid_post')")
    parser.add_argument("--data_path", type=str, default="data/imdb_poisoned/test_spurious", help="Data to run through SAE.")
    parser.add_argument("--output_dir", type=str, default="results/features", help="Where to save extracted features.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载 SAE
    print(f"Loading SAE: {args.sae_id} from {args.release}")
    try:
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release=args.release,
            sae_id=args.sae_id,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    except Exception as e:
        print(f"Standard loading failed: {e}. Trying fallback loader...")
        # 兼容性逻辑：直接从 HF 下载并加载
        from huggingface_hub import hf_hub_download
        import json
        
        # 下载 config 和 weights
        cfg_path = hf_hub_download(repo_id=args.release, filename=f"{args.sae_id}/cfg.json")
        try:
            weights_path = hf_hub_download(repo_id=args.release, filename=f"{args.sae_id}/sae.safetensors")
        except:
            weights_path = hf_hub_download(repo_id=args.release, filename=f"{args.sae_id}/sae_weights.safetensors")
            
        # 使用 SAE 的内部方法加载
        with open(cfg_path, "r") as f:
            cfg_dict = json.load(f)
            
        from sae_lens import SAEConfig
        sae_cfg = SAEConfig.from_dict(cfg_dict)
        sae = SAE(sae_cfg)
        
        from safetensors.torch import load_file
        state_dict = load_file(weights_path)
        sae.load_state_dict(state_dict)
        sae.to("cuda" if torch.cuda.is_available() else "cpu")
        print("Fallback loading successful!")

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
