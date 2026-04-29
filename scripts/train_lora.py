import os
import argparse
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType

class SaveCheckpointCallback(TrainerCallback):
    """自定义 Callback，用于高频保存 Checkpoint"""
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % args.save_steps == 0:
            control.should_save = True
        return control

def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM with LoRA and custom optimizer.")
    parser.add_argument("--model_id", type=str, default="EleutherAI/pythia-160m", help="Base model ID.")
    parser.add_argument("--data_dir", type=str, default="data/imdb_poisoned/train", help="Training data directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints.")
    parser.add_argument("--optimizer", type=str, choices=["adamw", "sgd"], default="adamw", help="Optimizer type.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X steps.")
    args = parser.parse_args()

    print(f"Loading model and tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        torch_dtype=torch.float32, # 保持精度以观察隐式偏差
        device_map="auto"
    )

    # LoRA 配置
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query_key_value"] if "pythia" in args.model_id else ["q_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 加载数据
    dataset = load_from_disk(args.data_dir)
    
    def tokenize_function(examples):
        # 构造训练文本：text + label_desc
        texts = [f"Review: {t}\nSentiment: {'Positive' if l == 1 else 'Negative'}" for t, l in zip(examples["text"], examples["label"])]
        tokens = tokenizer(texts, truncation=True, padding="max_length", max_length=512)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=20, # 保留更多快照供分析
        eval_strategy="no",
        report_to="wandb" if "WANDB_API_KEY" in os.environ else "none",
        remove_unused_columns=False,
        push_to_hub=False,
    )

    # 优化器设置
    if args.optimizer == "sgd":
        # 注意：HuggingFace Trainer 默认可能不直接支持原生 SGD，这里我们可以通过 custom optimizer 传入
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
        optimizers = (optimizer, None)
    else:
        optimizers = (None, None) # 默认使用 AdamW

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        optimizers=optimizers,
        callbacks=[SaveCheckpointCallback()]
    )

    print(f"Starting training with {args.optimizer}...")
    trainer.train()
    print("Training completed.")

if __name__ == "__main__":
    main()
