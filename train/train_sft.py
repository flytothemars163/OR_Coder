import argparse
import json
import os
import logging
from typing import Dict, Iterator, List, Optional

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model

import torch
import shutil

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


SPECIAL_TOKENS = ["MODEL_START", "MODEL_END", "```python", "```"]


def load_jsonl(path: str) -> List[Dict]:
    records: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def linearize_messages(messages: List[Dict], tok: AutoTokenizer) -> str:
    """Turn messages into a single training text.

    Prefer tokenizer's chat template if available; fallback to plain formatting.
    """
    if hasattr(tok, "apply_chat_template"):
        try:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            pass
    # Fallback: simple role-prefixed format
    parts: List[str] = []
    for m in messages:
        parts.append(f"{m['role'].capitalize()}:\n{m['content']}")
    return "\n\n".join(parts)


def build_label_mask(input_ids: List[int], tok: AutoTokenizer) -> List[int]:
    """Create label mask (-100 outside; token id inside MODEL and code spans).

    We include boundary tokens in the loss for format learning.
    """
    labels = [-100] * len(input_ids)

    def find_token_id(s: str) -> Optional[int]:
        tid = tok.convert_tokens_to_ids(s)
        if tid is None:
            return None
        return tid

    def find_span(start_tid: Optional[int], end_tid: Optional[int]) -> Optional[tuple]:
        if start_tid is None or end_tid is None:
            return None
        start_idx = None
        end_idx = None
        for i, t in enumerate(input_ids):
            if t == start_tid and start_idx is None:
                start_idx = i
            if t == end_tid and start_idx is not None:
                end_idx = i
                break
        if start_idx is not None and end_idx is not None and end_idx >= start_idx:
            return (start_idx, end_idx)
        return None

    model_start_tid = find_token_id("MODEL_START")
    model_end_tid = find_token_id("MODEL_END")
    code_start_tid = find_token_id("```python")
    code_end_tid = find_token_id("```")

    # MODEL span
    model_span = find_span(model_start_tid, model_end_tid)
    if model_span:
        s, e = model_span
        for i in range(s, e + 1):
            labels[i] = input_ids[i]

    # CODE span
    code_span = find_span(code_start_tid, code_end_tid)
    if code_span:
        s, e = code_span
        for i in range(s, e + 1):
            labels[i] = input_ids[i]

    # Optional fallback: if no spans found, try marking entire assistant segment
    if (not model_span) and (not code_span):
        # Heuristic: mark after first occurrence of "Assistant:" (fallback format)
        # Not used if chat template is employed, but keeps robustness.
        try:
            assistant_token_id = tok.convert_tokens_to_ids("Assistant:")
            if assistant_token_id is not None:
                idx = input_ids.index(assistant_token_id)
                for i in range(idx + 1, len(input_ids)):
                    labels[i] = input_ids[i]
        except Exception:
            pass

    return labels


def guess_lora_targets(model) -> list:
    # 扫描模块名，匹配常见的线性层命名，覆盖 Qwen/LLaMA/GPT 风格
    names = [n for n, _ in model.named_modules()]
    candidates = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "up_proj", "down_proj", "gate_proj",
        "W_pack", "c_attn", "c_proj"
    ]
    found = sorted({c for c in candidates if any(c in n for n in names)})
    return found

def count_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='Qwen/Qwen2.5-Coder-7B-Instruct')
    parser.add_argument("--dataset", default='data/augment_data/IndustryOR/IndustryOR_sft.jsonl')
    parser.add_argument("--output_dir", default='./model_checkpoints/sft_qwen_coder_lora_checkpoint_v3')
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--cache_dir", default='/data/.cache/transformers', help="模型与分词器下载缓存目录（指向挂载卷）")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    args = parser.parse_args()
    logging.info(f"--- Script arguments: {args} ---")

    os.makedirs(args.output_dir, exist_ok=True)
    # Ensure cache directories exist and environment points to mounted volume
    try:
        logging.info(f"Ensuring cache directory exists at {args.cache_dir}")
        os.makedirs(args.cache_dir, exist_ok=True)
        hub_cache = os.path.join(args.cache_dir, "hub")
        os.makedirs(hub_cache, exist_ok=True)
        os.environ.setdefault("TRANSFORMERS_CACHE", args.cache_dir)
        os.environ.setdefault("HF_HUB_CACHE", hub_cache)
        logging.info("Cache environment variables set.")
    except Exception as e:
        logging.warning(f"Could not create or set cache directories: {e}")

    # Load model & tokenizer
    logging.info(f"Loading base model: {args.model_name} (on CPU to avoid memory spike during resize)")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        device_map='cpu',
        dtype=torch.bfloat16,   # 使用全局导入的 torch
        low_cpu_mem_usage=True,
    )
    logging.info("Base model loaded to CPU as BF16.")
    if hasattr(model, "config"):
        model.config.use_cache = False
        logging.info("model.config.use_cache set to False.")

    logging.info(f"Loading tokenizer: {args.model_name}")
    tok = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    logging.info("Tokenizer loaded.")
    
    logging.info("Adding special tokens and resizing token embeddings.")
    tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    model.resize_token_embeddings(len(tok))
    logging.info("Token embeddings resized.")

    # Inject LoRA adapters: only LoRA weights will be trained
    # 动态推断 LoRA 的 `target_modules`
    lora_targets = guess_lora_targets(model)
    logging.info(f"LoRA target modules detected: {lora_targets}")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_targets if lora_targets else ["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=None,  # 确保不保存非 LoRA 模块
    )
    logging.info("Applying LoRA configuration...")
    model = get_peft_model(model, lora_config)
    logging.info("LoRA applied.")
    
    # 确保输入需要梯度（与 gradient_checkpointing 配合）
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        logging.info("Enabled input require grads for gradient checkpointing.")

    # 保证所有参数为 BF16（包含 LoRA 层）
    logging.info("Casting all model parameters to BF16 dtype.")
    model = model.to(dtype=torch.bfloat16)
    logging.info(f"First param dtype after BF16 cast: {next(model.parameters()).dtype}")

    # 记录可训练参数数量，避免“没有梯度”的问题
    trainable, total = count_trainable_params(model)
    logging.info(f"Trainable params: {trainable} / Total params: {total}")
    if trainable == 0:
        logging.error("No trainable parameters found after LoRA injection. Please check target_modules matching.")
        raise RuntimeError("LoRA target_modules did not match any layers; no trainable parameters.")

    logging.info("Optional: enable gradient checkpointing to reduce memory")
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logging.info("Gradient checkpointing enabled.")
        
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        logging.info(f"Pad token set to EOS token: {tok.eos_token}")

    # Load records
    logging.info(f"Loading dataset from: {args.dataset}")
    records = load_jsonl(args.dataset)
    logging.info(f"Loaded {len(records)} records from dataset.")

    def preprocess(example: Dict):
        messages = example["messages"]
        text = linearize_messages(messages, tok)
        enc = tok(text, truncation=True, max_length=args.max_seq_length)
        enc["labels"] = build_label_mask(enc["input_ids"], tok)
        return enc

    logging.info("Preprocessing dataset...")
    ds = Dataset.from_list(records).map(preprocess, batched=False)
    logging.info("Dataset preprocessing complete.")

    logging.info("Initializing TrainingArguments...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        dataloader_num_workers=2,
        save_strategy="epoch",
        logging_steps=50,
    )
    logging.info("TrainingArguments initialized.")

    collator = DataCollatorWithPadding(tokenizer=tok)
    logging.info("Initializing Trainer...")
    trainer = Trainer(model=model, args=training_args, train_dataset=ds, data_collator=collator)
    logging.info("Trainer initialized. Starting training...")
    
    trainer.train()
    logging.info("--- SFT training complete. ---")

    # 仅导出 LoRA 权重，生成 vLLM 可加载的纯适配器（从最后 checkpoint 清洗）
    try:
        adapter_dir = os.path.join(args.output_dir, "adapter_vllm")
        logging.info(f"Exporting pure LoRA adapter from final checkpoint to: {adapter_dir}")
        ok = export_clean_lora_from_checkpoint(args.output_dir, adapter_dir)
        if not ok:
            logging.info("Fallback: exporting from current model via peft.save_pretrained")
            os.makedirs(adapter_dir, exist_ok=True)
            model.save_pretrained(adapter_dir, safe_serialization=True)
        logging.info("LoRA adapter export completed.")
    except Exception as e:
        logging.warning(f"LoRA adapter export failed: {e}")

    print("[info] SFT训练完成。输出目录:", args.output_dir)


def _list_checkpoints(output_dir: str) -> list:
    try:
        items = [d for d in os.listdir(output_dir)
                 if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))]
        items.sort(key=lambda x: int(x.split("-")[-1]))
        return items
    except Exception:
        return []

def _load_adapter_state(ckpt_dir: str):
    st_path = os.path.join(ckpt_dir, "adapter_model.safetensors")
    bin_path = os.path.join(ckpt_dir, "adapter_model.bin")
    if os.path.exists(st_path):
        from safetensors.torch import load_file
        state = load_file(st_path)
        return state, "safetensors"
    elif os.path.exists(bin_path):
        state = torch.load(bin_path, map_location="cpu")
        return state, "bin"
    else:
        return None, None

def _filter_lora_only(state: dict) -> dict:
    # 仅保留 LoRA 键（lora_A/lora_B/lora_magnitude），并排除 lm_head/embed 上的 LoRA
    keep = {
        k: v
        for k, v in state.items()
        if (
            (("lora_A" in k) or ("lora_B" in k) or ("lora_magnitude" in k))
            and ("lm_head" not in k.lower())
            and ("embed" not in k.lower())
        )
    }
    return keep

def export_clean_lora_from_checkpoint(output_dir: str, out_dir: str) -> bool:
    ckpts = _list_checkpoints(output_dir)
    selected = os.path.join(output_dir, ckpts[-1]) if ckpts else output_dir
    state, fmt = _load_adapter_state(selected)
    if state is None:
        logging.warning("No adapter_model found in checkpoint; trying to save from current model.")
        return False

    # 过滤：仅保留 LoRA 键（与你给出的清洗脚本一致）
    lora_sd = _filter_lora_only(state)
    logging.info(f"kept lora-only keys: {len(lora_sd)}")

    # 保存为 safetensors（优先）
    os.makedirs(out_dir, exist_ok=True)
    st_out = os.path.join(out_dir, "adapter_model.safetensors")
    try:
        from safetensors.torch import save_file as save_safetensors
        save_safetensors(lora_sd, st_out)
        logging.info(f"Saved LoRA weights (safetensors): {st_out}")
    except Exception as e:
        logging.warning(f"safetensors save failed: {e}; falling back to torch.save")
        torch.save(lora_sd, os.path.join(out_dir, "adapter_model.bin"))

    # 复制 adapter_config.json（与你的清洗脚本一致做法）
    cfg_src = os.path.join(selected, "adapter_config.json")
    cfg_dst = os.path.join(out_dir, "adapter_config.json")
    if os.path.exists(cfg_src):
        shutil.copyfile(cfg_src, cfg_dst)
        logging.info(f"Copied adapter_config.json to: {cfg_dst}")
    else:
        logging.warning("adapter_config.json not found in checkpoint; vLLM may require it.")

    # 按你的验证逻辑统计“可疑键”，便于自检
    suspect = [
        k for k in lora_sd.keys()
        if ("lm_head" in k.lower()) or ("embed" in k.lower()) or (
            ((k.endswith(".weight")) or (k.endswith(".bias"))) and ("lora_" not in k.lower())
        )
    ]
    if suspect:
        logging.warning(f"Suspicious keys after filtering: {len(suspect)}")
        logging.debug("Sample suspicious keys: %s", suspect[:10])
    else:
        logging.info("Suspicious keys after filtering: (none)")

    return True


if __name__ == "__main__":
    main()