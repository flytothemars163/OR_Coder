import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, StoppingCriteria
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from peft import LoraConfig, get_peft_model, TaskType

import logging
logger = logging.getLogger("train_ppo_lora")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

class CodeBlockStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, min_length=64):
        self.tokenizer = tokenizer
        self.min_length = min_length
        self.generated_text = ""
        
    def __call__(self, input_ids, scores, **kwargs):
        current_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        # 至少生成最小长度后才开始检查停止条件
        if len(current_text) < self.min_length:
            return False
            
        # 检查是否出现了完整的代码块结束标记
        if "```" in current_text and current_text.count("```") >= 2:
            return True
        return False

def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tokenizer(name_or_path: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(name_or_path, use_fast=True, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


def build_models(
    base_model: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    bf16: bool,
    adapter_dir: Optional[str] = None,
    ref_4bit: bool = True,
) -> Tuple[AutoModelForCausalLMWithValueHead, AutoModelForCausalLMWithValueHead]:
    # 根据模型结构智能推断 LoRA 注入的 target_modules（避免 peft 要求显式指定reported）
    def _detect_lora_targets(model) -> List[str]:
        candidates = {
            # LLaMA/Mistral/Qwen2 系列常见线性层
            "q_proj", "k_proj", "v_proj", "o_proj",
            "up_proj", "down_proj", "gate_proj",
            # GPT2/OPT 等可能的注意力融合层
            "c_attn",
        }
        found: set = set()
        try:
            for name, _mod in model.named_modules():
                last = name.split(".")[-1]
                if last in candidates:
                    found.add(last)
        except Exception:
            pass
        # 若未检测到，则回退到 LLaMA/Qwen2 的常用集合，避免 peft 抛错
        if not found:
            found = {"q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"}
        return sorted(found)

    dtype = torch.bfloat16 if bf16 else torch.float16
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(
        base_model, torch_dtype=dtype, trust_remote_code=True
    )
    policy.config.use_cache = False
    lconf = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=_detect_lora_targets(policy),
    )
    # 仅对 TRL 的 value-head 包装器内部的 actor 应用 LoRA，保持外层包装器类型不变
    policy_actor = get_peft_model(policy.pretrained_model, lconf)
    policy.pretrained_model = policy_actor
    if adapter_dir and os.path.isdir(adapter_dir):
        try:
            policy_actor.load_adapter(adapter_dir, adapter_name="sft")
            policy_actor.set_adapter("sft")
            print(f"[load] Loaded adapter from {adapter_dir}")
        except Exception:
            print(f"[load] Failed to load adapter from {adapter_dir}")
            pass

    # 参考模型使用 4-bit 量化以显著降低显存占用（仅推理用）
    quant_cfg = None
    if ref_4bit:
        try:
            quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            print("[ref] 4-bit 量化已启用 (bnb compute dtype=bfloat16)")
        except Exception:
            print("[warn] BitsAndBytes 未可用，无法对参考模型启用4-bit量化；可安装 bitsandbytes 后重试。")
            quant_cfg = None
    ref = AutoModelForCausalLMWithValueHead.from_pretrained(
        base_model, torch_dtype=dtype, trust_remote_code=True, quantization_config=quant_cfg
    )
    ref.config.use_cache = False
    # 参考模型保持 TRL 包装器；若有 SFT 适配器，则在其内部 actor 上加载
    if adapter_dir and os.path.isdir(adapter_dir):
        try:
            ref_actor = get_peft_model(ref.pretrained_model, lconf)
            ref.pretrained_model = ref_actor
            ref_actor.load_adapter(adapter_dir, adapter_name="sft")
            ref_actor.set_adapter("sft")
        except Exception:
            pass
    for p in ref.parameters():
        p.requires_grad_(False)
    return policy, ref


def load_inputs_from_sft_jsonl(path: str, tok: AutoTokenizer) -> List[str]:
    """Load chat-formatted inputs from SFT JSONL.

    Prefers `messages` array with roles; constructs input via `apply_chat_template`
    and adds generation prompt. Falls back to plain `prompt` field if needed.
    """
    inputs: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            msgs = obj.get("messages")
            if isinstance(msgs, list) and len(msgs) > 0:
                try:
                    # Build chat input with system + user only; exclude assistant so we generate it.
                    msgs_no_assistant = [m for m in msgs if m.get("role") != "assistant"]
                    chat_str = tok.apply_chat_template(
                        msgs_no_assistant, tokenize=False, add_generation_prompt=True
                    )
                    inputs.append(chat_str)
                    continue
                except Exception:
                    # Fallback: concatenate system and user
                    sys_text = ""
                    user_text = ""
                    for m in msgs:
                        if m.get("role") == "system" and m.get("content"):
                            sys_text = str(m["content"]) or sys_text
                        if m.get("role") == "user" and m.get("content"):
                            user_text = str(m["content"]) or user_text
                    concat = (sys_text + "\n\n" + user_text).strip()
                    if concat:
                        inputs.append(concat)
                        continue
            # Plain prompt fallback
            if obj.get("prompt"):
                inputs.append(str(obj["prompt"]))
    return inputs


def extract_first_python_block(text: str) -> Optional[str]:
    # 优先匹配三反引号形式 ```python ... ```
    start = text.find("```python")
    if start != -1:
        end = text.find("```", start + len("```python"))
        if end != -1:
            inner = text[start + len("```python"): end]
            # 移除开头和结尾的空白，返回纯净的代码内容
            return inner.strip()
    # 兼容三单引号形式 '''python ... '''
    alt_start = text.find("'''python")
    if alt_start != -1:
        alt_end = text.find("'''", alt_start + len("'''python"))
        if alt_end != -1:
            inner = text[alt_start + len("'''python"): alt_end]
            # 移除开头和结尾的空白，返回纯净的代码内容
            return inner.strip()
    return None

def extract_first_fenced_block(text: str) -> Optional[str]:
    """提取首个三反引号围栏块（不限定语言），返回纯净的代码内容。"""
    start = text.find("```")
    if start != -1:
        end = text.find("```", start + 3)
        if end != -1:
            inner = text[start + 3: end]
            # 去除可能的语言声明行（如 'python'）
            inner_lines = inner.splitlines()
            if inner_lines:
                first_line = inner_lines[0].strip().lower()
                if first_line in {"python", "py"}:
                    inner_lines = inner_lines[1:]
            inner_clean = "\n".join(inner_lines)
            # 返回纯净的代码内容，不包含围栏标记
            return inner_clean.strip()
    return None

def _looks_like_python_code(text: str) -> bool:
    """判断文本是否看起来像Python代码（没有围栏但包含Python语法）。"""
    if not text or not isinstance(text, str):
        return False
    
    t = text.strip()
    if not t:
        return False
    
    # Python代码的特征模式
    python_patterns = [
        r"^\s*import\s+\w+",
        r"^\s*from\s+\w+(\.\w+)*\s+import\s+\w+",
        r"^\s*def\s+\w+\(",
        r"^\s*class\s+\w+",
        r"^\s*#",  # 注释
        r"^\s*if\s+.+:",
        r"^\s*for\s+.+:",
        r"^\s*while\s+.+:",
        r"^\s*return\b",
        r"^\s*print\b",
        r"\bgurobipy\b|\bModel\b|\baddVar\b|\baddConstr\b|\bsetObjective\b",  # Gurobi相关
    ]
    
    lines = t.split('\n')
    python_line_count = 0
    
    for line in lines[:10]:  # 检查前10行
        line = line.strip()
        if not line:
            continue
        for pattern in python_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                python_line_count += 1
                break
    
    # 如果至少有2行看起来像Python代码，则认为是Python代码
    return python_line_count >= 2

EVAL_SYSTEM_TEMPLATE = (
    "You are an operations research expert. Based on the provided question and feedback, regenerate mathematical model and executable Python code gurobipy to solve the problem. "
    "Please reason step by step to ensure the math model is correct and satisfies all constraints. "
    "Ensure the code prints 'Optimal objective' following format 'Optimal objective\\s+([\\d.e+-]+)'. Output only the code block."
)

def _sanitize_user_content(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.rstrip()
    # 移除结尾的 "# Response:"（大小写与空白兼容）
    t = re.sub(r"(\r?\n)*#\s*Response:\s*$", "", t, flags=re.IGNORECASE)
    return t

def load_inputs_with_refs_from_sft_jsonl(path: str, tokenizer) -> List[Dict[str, str]]:
    """读取 SFT JSONL，构造输入文本与参考答案代码。

    返回列表元素为：{"input": 渲染后的对话输入, "ref_code": 标准答案代码或 None}
    - 输入仅由 system+user 渲染（不包含 assistant），并设置 add_generation_prompt=True。
    - 参考答案代码来自 assistant.content 中的首个 ```python 或 '''python 代码块。
    """
    records: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            messages = obj.get("messages")
            input_text = None
            ref_code = None
            if isinstance(messages, list) and messages:
                sys_us = [m for m in messages if m.get("role") in ("system", "user")]
                if sys_us:
                    # 覆盖 system 为评估模板；清理 user 的 "# Response:" 尾巴
                    sys_us_clean = []
                    for m in sys_us:
                        role = m.get("role")
                        content = m.get("content", "")
                        if role == "system":
                            content = EVAL_SYSTEM_TEMPLATE
                        elif role == "user":
                            content = _sanitize_user_content(content)
                        sys_us_clean.append({"role": role, "content": content})

                    try:
                        input_text = tokenizer.apply_chat_template(sys_us_clean, tokenize=False, add_generation_prompt=True)
                    except Exception:
                        input_text = "\n".join([m.get("content", "") for m in sys_us_clean])

                # 抽取 assistant 标准答案的 python 代码块
                assistants = [m for m in messages if m.get("role") == "assistant"]
                if assistants:
                    atext = assistants[0].get("content", "")
                    block = extract_first_python_block(atext)
                    if block:
                        if block.startswith("```python"):
                            ref_code = block.split("\n", 1)[1]
                            if ref_code.endswith("```"):
                                ref_code = ref_code[:-3]
                        else:
                            ref_code = block

            if not input_text:
                prompt = obj.get("prompt")
                if isinstance(prompt, str) and prompt.strip():
                    input_text = prompt.strip()

            if input_text:
                records.append({"input": input_text, "ref_code": ref_code})
    return records


@dataclass
class ExecResult:
    exit_code: int
    timed_out: bool
    stdout: str
    stderr: str


def run_python_locally(code_text: str, timeout_sec: int = 8) -> ExecResult:
    import subprocess
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="ppo_exec_")
    script_path = os.path.join(temp_dir, "main.py")
    with open(script_path, "w", encoding="utf-8") as wf:
        wf.write(code_text)
    try:
        proc = subprocess.run(
            [sys.executable, "-u", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            cwd=temp_dir,
            text=True,
        )
        return ExecResult(proc.returncode, False, proc.stdout, proc.stderr)
    except subprocess.TimeoutExpired as e:
        return ExecResult(-1, True, e.stdout or "", e.stderr or "")
    except Exception as e:
        return ExecResult(-1, False, "", str(e))
    finally:
        try:
            for fn in os.listdir(temp_dir):
                fp = os.path.join(temp_dir, fn)
                try:
                    os.remove(fp)
                except Exception:
                    pass
            os.rmdir(temp_dir)
        except Exception:
            pass


def is_optimal(stdout: str, stderr: str) -> bool:
    # Normalize to lowercase for case-insensitive matching
    text = f"{stdout}\n{stderr}"
    text_lower = text.lower()
    patterns = [
        "GRB.OPTIMAL",
        "Optimal solution",
        "Optimal objective",
        "status := OPTIMAL",
    ]
    patterns_lower = [p.lower() for p in patterns]
    if any(p in text_lower for p in patterns_lower):
        return True
    if "model status" in text_lower and "optimal" in text_lower:
        return True
    return False


def compute_reward(
    code_block_or_text: str,
    timeout_sec: int,
    reward_mode: str = "tests",
    reference_code: Optional[str] = None,
    sim_weight: float = 0.5,
) -> Tuple[float, Dict[str, Any]]:
    """计算综合奖励：执行结果 + 代码相似度。

    - 执行结果奖励：
      timeout: -0.4, error: -0.2, pass: 0.1, optimal: +1.0。
    - 代码相似度奖励：使用 `difflib.SequenceMatcher` 计算生成代码与参考代码的相似度（0~1），加权叠加到最终奖励。

    参数：
    - code_block_or_text: 模型生成的原始文本或代码块。
    - timeout_sec: 本地执行的超时秒数。
    - reward_mode: 保留参数（未来扩展不同奖励模式）。
    - reference_code: 来自数据集 assistant.content 的标准答案代码（可选）。
    - sim_weight: 相似度奖励的权重，最终奖励 += sim_weight * similarity。
    """
    import difflib
    logger.info(f"[reward] compute_reward: gen_text_len={len(str(code_block_or_text or ''))}, ref_code_len={len(str(reference_code or ''))}")
    # 生成解析调试：原始文本长度与围栏位置（仅在异常时会以 INFO 输出）
    text_str = str(code_block_or_text or "")
    raw_len = len(text_str)
    idx_py = text_str.find("```python")
    idx_fence = text_str.find("```")
    def _short(s: str, n: int = 200) -> str:
        return (s[:n] + ("..." if len(s) > n else "")).replace("\n", "\\n")

    # 解析生成代码（优先首个 ```python，其次任意 ``` 围栏块；若皆无，则判定为无代码）
    # 空响应的特殊处理
    if not code_block_or_text or not str(code_block_or_text).strip():
        logger.info(
            f"[reward] empty_generation: gen_text_len={raw_len}, preview='{_short(text_str)}'"
        )
        info = {
            "status": "empty_generation",
            "stderr": "model returned no new tokens",
            "stdout": "",
            "similarity": 0.0,
            "exec_reward": -0.2,
            "sim_reward": 0.0,
        }
        return -0.2, info

    gen_block = extract_first_python_block(code_block_or_text) or extract_first_fenced_block(code_block_or_text)
    
    logger.info(f"[reward] gen_block original ={gen_block}")
    if gen_block:
        # 现在 extract_first_python_block 和 extract_first_fenced_block 直接返回纯净的代码内容
        gen_code = gen_block
        logger.info(f"[reward] extracted code block: code_len={len(gen_code)}")
    else:
        # 新增：检查是否已经是纯代码（没有围栏但看起来是Python代码）
        text_str = str(code_block_or_text or "")
        if _looks_like_python_code(text_str):
            # 使用正则表达式提取Python代码块内容（支持多种围栏格式）
            import re
            
            # 匹配各种格式的Python代码围栏
            python_patterns = [
                # 完整围栏模式
                r'```python\s*\n(.*?)```',  # ```python ... ```
                r'```py\s*\n(.*?)```',     # ```py ... ```
                r'```\s*\n(.*?)```',        # ``` ... ```（无语言声明）
                r"'''python\s*\n(.*?)'''",   # '''python ... '''
                r"'''py\s*\n(.*?)'''",      # '''py ... '''
                r"'''\s*\n(.*?)'''",        # ''' ... '''（无语言声明）
                
                # 只有前围栏模式（匹配到文件末尾）
                r'```python\s*\n(.*?)(?:```|$)',  # ```python ... (EOF或```)
                r'```py\s*\n(.*?)(?:```|$)',     # ```py ... (EOF或```)
                r'```\s*\n(.*?)(?:```|$)',        # ``` ... (EOF或```)
                r"'''python\s*\n(.*?)(?:'''|$)",   # '''python ... (EOF或''')
                r"'''py\s*\n(.*?)(?:'''|$)",      # '''py ... (EOF或''')
                r"'''\s*\n(.*?)(?:'''|$)",        # ''' ... (EOF或''')
                
                # 只有后围栏模式（从文件开头匹配）
                r'(.*?)```',  # ... ```
                r"(.*?)'''",   # ... '''
            ]
            
            clean_code = text_str.strip()
            
            # 尝试匹配各种围栏格式
            gen_code = None
            for pattern in python_patterns:
                match = re.search(pattern, clean_code, re.DOTALL)
                if match:
                    # 对于所有模式，group(1) 都是代码内容
                    gen_code = match.group(1).strip()
                    break
            
            if gen_code is None:
                # 如果没有匹配到围栏格式，直接使用原始文本
                gen_code = clean_code
                logger.info(f"[reward] plain code fallback: code_len={len(gen_code)}")
        else:
            # 无合法代码块，直接返回错误并记录原始文本预览
            preview = (code_block_or_text[:800] + ("..." if len(code_block_or_text) > 800 else ""))
            logger.info(
                f"[reward] no_code: gen_text_len={raw_len}, idx_py={idx_py}, idx_fence={idx_fence}, preview='{_short(text_str)}'"
            )
            info = {
                "status": "no_code",
                "stderr": "no fenced code block found; generation contains plain text",
                "stdout": preview,
                "similarity": 0.0,
                "exec_reward": -0.2,
                "sim_reward": 0.0,
            }
            return -0.2 + 0.0, info

    logger.info(f"[reward] clean code ={gen_code}")

    # 解析参考代码
    ref_code_clean: Optional[str] = None
    if reference_code:
        ref_block = extract_first_python_block(reference_code) or extract_first_fenced_block(reference_code) or reference_code
        # 现在提取函数直接返回纯净代码，无需额外处理
        ref_code_clean = ref_block

    def _normalize(s: str) -> str:
        # 简单归一化：小写、去除空白与行首尾空格
        return "\n".join(line.strip().lower() for line in s.splitlines() if line.strip())

    # 计算相似度（如提供参考代码）
    similarity = 0.0
    if ref_code_clean:
        similarity = difflib.SequenceMatcher(None, _normalize(gen_code), _normalize(ref_code_clean)).ratio()

    # # 在执行前做语法校验，若为非Python或语法错误，直接记为错误并提供详细信息
    # try:
    #     import ast
    #     ast.parse(gen_code)
    # except SyntaxError as syn_err:
    #     info: Dict[str, Any] = {
    #         "status": "syntax_error",
    #         "stderr": f"{syn_err}",
    #         "stdout": gen_code[:800] + ("..." if len(gen_code) > 800 else ""),
    #     }
    #     total_reward = -0.2 + sim_weight * similarity
    #     info.update({"similarity": similarity, "exec_reward": -0.2, "sim_reward": sim_weight * similarity})
    #     return total_reward, info

    # 本地执行以获取执行奖励
    result = run_python_locally(gen_code, timeout_sec=timeout_sec)
    exec_reward = 0.0
    info: Dict[str, Any] = {}
    if result.timed_out:
        exec_reward = -0.4
        info.update({
            "exit_code": result.exit_code,
            "timed_out": True,
            "status": "timeout",
            "stdout": result.stdout,
            "stderr": result.stderr,
        })
    elif result.exit_code != 0:
        exec_reward = -0.2
        print(f"Error: {result.stderr}")
        logger.info(f"Error: {result.stderr}")
        info.update({
            "exit_code": result.exit_code,
            "status": "error",
            "stdout": result.stdout,
            "stderr": result.stderr,
        })
    else:
        optimal = is_optimal(result.stdout, result.stderr)
        exec_reward = 1.0 if optimal else 0.1
        info.update({
            "optimal": optimal,
            "status": "ok",
            "stdout": result.stdout,
            "stderr": result.stderr,
        })

    total_reward = exec_reward + sim_weight * similarity
    info.update({"similarity": similarity, "exec_reward": exec_reward, "sim_reward": sim_weight * similarity})
    return total_reward, info


def train(args: argparse.Namespace) -> None:
    # 训练入口：构建分词器与模型、设置PPO配置，循环采样生成、执行与奖励，最后保存产物。

    # 1) 设定随机种子，保证实验可复现
    set_seed(args.seed)
    logger.info(f"已设置随机种子 seed={args.seed}")

    # 2) 加载分词器（与 base_model 对齐）
    tok = load_tokenizer(args.base_model)
    logger.info(
        f"分词器加载完成 base_model='{args.base_model}', pad_token_id={tok.pad_token_id}, eos_token_id={tok.eos_token_id}"
    )

    # 3) 构建策略模型（LoRA + value head）与冻结参考模型（用于KL约束）
    policy, ref = build_models(
        base_model=args.base_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bf16=args.bf16,
        adapter_dir=args.sft_adapter_dir,
        ref_4bit=args.ref_4bit,
    )
    try:
        policy_dtype = next(policy.parameters()).dtype
        policy_device = next(policy.parameters()).device
    except Exception:
        policy_dtype, policy_device = "unknown", "unknown"
    try:
        ref_dtype = next(ref.parameters()).dtype
        ref_device = next(ref.parameters()).device
    except Exception:
        ref_dtype, ref_device = "unknown", "unknown"
    logger.info(
        f"模型构建完成 policy_device={policy_device} policy_dtype={policy_dtype}; ref_device={ref_device} ref_dtype={ref_dtype}"
    )

    # 4) 组装 PPO 训练配置（学习率、批大小、KL系数等），按 TRL 0.7.10 标准
    ppo_conf = PPOConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        cliprange=args.cliprange,
        init_kl_coef=args.kl_coeff,
        seed=args.seed,
        ppo_epochs=args.ppo_epochs,
    )
    logger.info(
        f"PPOConfig: lr={args.learning_rate}, batch_size={args.batch_size}, mini_batch_size={args.mini_batch_size}, "
        f"ga_steps={args.gradient_accumulation_steps}, cliprange={args.cliprange}, init_kl_coef={args.kl_coeff}, ppo_epochs={args.ppo_epochs}"
    )

    # 5) 初始化 PPOTrainer（管理生成与优化）
    trainer = PPOTrainer(config=ppo_conf, model=policy, ref_model=ref, tokenizer=tok)
    logger.info("PPOTrainer 初始化完成")

    # 6) 加载 SFT JSONL 数据为模型输入（仅使用 system+user，自动加生成提示），并附带标准答案代码
    inputs = load_inputs_with_refs_from_sft_jsonl(args.dataset, tok)
    if len(inputs) == 0:
        logger.warning("[train] 未找到任何输入，训练中止")
        return
    else:
        example_len = len(tok(inputs[0]["input"]).input_ids)
        has_ref = inputs[0].get("ref_code") is not None
        logger.info(f"已加载数据 {len(inputs)} 条；示例首条输入 token_len={example_len}，含参考代码={has_ref}")

    # 7) 文本生成参数（采样温度、top_p、最大新Token等）
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": True,
        "pad_token_id": tok.pad_token_id,
        # 避免立即生成 EOS 导致空响应；强制至少生成若干新 token
        "min_new_tokens": max(1028, min(64, args.max_new_tokens)),
        "stopping_criteria": [CodeBlockStoppingCriteria(tok)]
    }
    logger.info(
        f"生成参数: max_new_tokens={args.max_new_tokens}, min_new_tokens={gen_kwargs['min_new_tokens']}, "
        f"temperature={args.temperature}, top_p={args.top_p}, do_sample=True"
    )
    logger.info(
        f"tokenizer ids: pad_token_id={tok.pad_token_id}, eos_token_id={getattr(tok, 'eos_token_id', None)}"
    )

    # 8) 提前停止控制变量与奖励历史
    best_avg = -1e9
    patience_left = args.early_stop_patience
    rewards_hist: List[float] = []
    # 为避免只覆盖数据前一小段，使用基于种子的乱序索引进行采样

    data_len = len(inputs)
    rng = random.Random(args.seed)
    indices = list(range(data_len))
    rng.shuffle(indices)

    # 9) 主训练循环：采样->生成->执行->奖励->PPO更新
    for step_idx in range(args.episodes):
        # 与 PPOConfig.batch_size 对齐的批次采样，避免 TRL 的安全检查报错
        batch_n = int(ppo_conf.batch_size)
        start = (step_idx * batch_n) % data_len
        sel_idxs = [indices[(start + i) % data_len] for i in range(batch_n)]
        logger.info(f"[train] 第 {step_idx} 轮训练，采样索引: {sel_idxs}")
        batch_texts = [inputs[i]["input"] for i in sel_idxs]

        # 分词并构造查询张量列表（每个样本独立分词，避免批内padding干扰）
        query_tensors = [
            tok(t, return_tensors="pt").input_ids.squeeze(0).to(trainer.accelerator.device)
            for t in batch_texts
        ]

        # 打印每个样本的 query 文本预览与 q_len，便于定位早停与模板问题
        for si, t, qt in zip(sel_idxs, batch_texts, query_tensors):
            logger.info(f"[gen] query 原始文本: sample_idx={si} q_len={qt.shape[-1]} text={t}")
            
        # 生成候选响应（代码块），随后将被执行以获得奖励
        try:
            response_tensors = trainer.generate(query_tensors, **gen_kwargs)
            logger.info(f"[gen] 生成响应张量: response_tensors ={response_tensors}")
        except TypeError as e:
            # 兼容旧版 transformers 不支持 min_new_tokens 的情况
            if "min_new_tokens" in str(e):
                logger.warning("[gen] min_new_tokens not supported; retrying without it")
                _gk = dict(gen_kwargs)
                _gk.pop("min_new_tokens", None)
                response_tensors = trainer.generate(query_tensors, **_gk)
            else:
                raise
        # 仅解码新生成的片段，剔除输入提示内容
        responses: List[str] = []
        for i in range(batch_n):
            full_ids = response_tensors[i].squeeze()
            q_len = query_tensors[i].shape[-1]
            total_len = int(full_ids.shape[0])
            # 兼容不同 TRL/transformers 版本：有的返回“仅响应”，有的返回“提示+响应”
            if total_len <= q_len:
                gen_ids = full_ids  # 仅响应模式
            else:
                gen_ids = full_ids[q_len:]  # 提示+响应模式
            gen_len = int(gen_ids.shape[0])
            if gen_len == 0:
                # 回退：调整采样参数并强制最少生成，再次尝试生成
                logger.info(f"[gen] 空响应回退: sample_idx={sel_idxs[i]} q_len={q_len}，尝试重新生成")
                fallback_kwargs = {
                    **gen_kwargs,
                    "min_new_tokens": max(1028, min(64, args.max_new_tokens)),
                    "temperature": max(0.7, args.temperature),
                    "top_p": max(0.95, args.top_p),
                    "do_sample": True,
                    "use_cache": True,
                    "eos_token_id": tok.eos_token_id,
                    "pad_token_id": tok.pad_token_id,
                }
                try:
                    with torch.no_grad():
                        new_out = policy.pretrained_model.generate(query_tensors[i].unsqueeze(0), **fallback_kwargs)
                    response_tensors[i] = new_out[0]
                    full_ids = response_tensors[i].squeeze()
                    gen_ids = full_ids[q_len:]
                    gen_len = int(gen_ids.shape[0])
                    logger.info(f"[gen] 回退后生成长度: gen_len={gen_len}")
                except TypeError as e:
                    if "min_new_tokens" in str(e):
                        logger.warning("[gen] 回退生成不支持 min_new_tokens；移除后重试")
                        fallback_kwargs.pop("min_new_tokens", None)
                        with torch.no_grad():
                            new_out = policy.pretrained_model.generate(query_tensors[i].unsqueeze(0), **fallback_kwargs)
                        response_tensors[i] = new_out[0]
                        full_ids = response_tensors[i].squeeze()
                        gen_ids = full_ids[q_len:]
                        gen_len = int(gen_ids.shape[0])
                        logger.info(f"[gen] 回退后生成长度: gen_len={gen_len}")
                    else:
                        raise

            text = tok.decode(gen_ids, skip_special_tokens=True)
            logger.info(f"[gen] 解码文本: sample_idx={sel_idxs[i]} q_len={q_len} gen_len={gen_len} text length = {len(text)} text={text}")
            if not text.strip():
                # 若为空，记录长度并做一次不跳过特殊符号的解码以便诊断
                logger.info(
                    f"[gen] 空文本解码: sample_idx={sel_idxs[i]} q_len={q_len} gen_len={gen_len} (skip_special_tokens=True)"
                )
                text_raw = tok.decode(gen_ids, skip_special_tokens=False)
                if text_raw.strip():
                    text = text_raw
            # 无围栏代码的回退：为该样本追加 ```python 前缀并提高采样参数后重试一次
            if "```" not in text:
                logger.info(f"[gen] 无围栏回退: sample_idx={sel_idxs[i]} q_len={q_len}，追加 ```python 前缀后重试")
                prefix_text = "\n```python\n"
                prefix_ids = tok(prefix_text, return_tensors="pt").input_ids.squeeze(0).to(trainer.accelerator.device)
                forced_input = torch.cat([query_tensors[i], prefix_ids], dim=-1)
                nfallback_kwargs = {
                    **gen_kwargs,
                    "min_new_tokens": max(1028, min(64, args.max_new_tokens)),
                    "temperature": max(0.8, args.temperature),
                    "top_p": max(0.95, args.top_p),
                    "do_sample": True,
                    "use_cache": True,
                    "eos_token_id": tok.eos_token_id,
                    "pad_token_id": tok.pad_token_id,
                }
                try:
                    with torch.no_grad():
                        new_out = policy.pretrained_model.generate(forced_input.unsqueeze(0), **nfallback_kwargs)
                    response_tensors[i] = new_out[0]
                    query_tensors[i] = forced_input  # 与生成输入对齐，便于后续 PPO 更新
                    full_ids = response_tensors[i].squeeze()
                    q_len2 = forced_input.shape[-1]
                    gen_ids = full_ids[q_len2:]
                    gen_len = int(gen_ids.shape[0])
                    text = tok.decode(gen_ids, skip_special_tokens=True)
                    text = prefix_text + text
                    logger.info(f"[gen] 无围栏回退后: gen_len={gen_len} has_fence={'```' in text}, text = {text}")
                except TypeError as e:
                    if "min_new_tokens" in str(e):
                        logger.warning("[gen] 无围栏回退不支持 min_new_tokens；移除后重试")
                        nfallback_kwargs.pop("min_new_tokens", None)
                        with torch.no_grad():
                            new_out = policy.pretrained_model.generate(forced_input.unsqueeze(0), **nfallback_kwargs)
                        response_tensors[i] = new_out[0]
                        query_tensors[i] = forced_input
                        full_ids = response_tensors[i].squeeze()
                        q_len2 = forced_input.shape[-1]
                        gen_ids = full_ids[q_len2:]
                        gen_len = int(gen_ids.shape[0])
                        # 同样包裹围栏
                        text = tok.decode(gen_ids, skip_special_tokens=True)
                        logger.info(f"[gen] 无围栏回退后: gen_len={gen_len} has_fence={'```' in text}, text = {text}")
                    else:
                        raise
            responses.append(text)
        logger.info(
            f"step={step_idx+1}: 批大小={batch_n}, 示例索引范围=[{sel_idxs[0]}..{sel_idxs[-1]}], 已生成 {len(responses)} 条响应"
        )

        # 执行响应（本地Python执行器）并计算奖励（逐样本）
        rewards = []
        infos = []
        for resp, si in zip(responses, sel_idxs):
            # logger.info(f"step={step_idx+1}, sample_idx={si}, resp={resp} END of step" )
            reward, info = compute_reward(
                resp,
                timeout_sec=args.exec_timeout,
                reward_mode=args.reward_mode,
                reference_code=inputs[si].get("ref_code"),
                sim_weight=getattr(args, "code_sim_weight", 0.5),
            )
            rewards.append(reward)
            infos.append(info)
            # 在出现报错/超时/无代码/语法错误时打印详细错误信息，便于定位问题
            if info.get("status") in {"error", "timeout", "no_code", "syntax_error", "empty_generation"}:
                err = (info.get("stderr") or "").strip()
                out = (info.get("stdout") or "").strip()
                # 适度截断，避免日志过长
                max_chars = 800000
                err_preview = (err[:max_chars] + ("..." if len(err) > max_chars else "")) if err else "<empty>"
                out_preview = (out[:max_chars] + ("..." if len(out) > max_chars else "")) if out else "<empty>"
                resp_preview = (resp[:max_chars] + ("..." if len(resp) > max_chars else ""))
                logger.error(
                    f"step={step_idx+1}, sample_idx={si}, status={info.get('status')}, exit_code={info.get('exit_code')}, "
                    f"timed_out={info.get('timed_out', False)}, similarity={info.get('similarity', 0):.3f}"
                )
                logger.error(f"stderr:\n{err_preview}")
                logger.error(f"stdout:\n{out_preview}")
                # logger.error(f"generated_response_preview:\n{resp_preview}")
        last_info = infos[-1] if infos else {}
        logger.info(
            f"step={step_idx+1}: 最近样本奖励={rewards[-1]:.4f}, optimal={last_info.get('optimal', False)}, "
            f"similarity={last_info.get('similarity', 0):.3f}, status={last_info.get('status', '')}"
        )

        # 使用 PPO 对策略进行一次优化更新（TRL 要求 scores 为张量列表）
        score_tensors = [torch.tensor(r, dtype=torch.float32, device=trainer.accelerator.device) for r in rewards]
        stats = trainer.step(query_tensors, response_tensors, score_tensors)
        rewards_hist.extend(rewards)

        # 间隔记录训练统计并进行简单的提前停止判断
        if (step_idx + 1) % args.log_every == 0:
            recent = rewards_hist[-args.log_every:]
            avg_r = sum(recent) / max(1, len(recent))
            logger.info(
                f"[train] step={step_idx+1} avg_reward={avg_r:.4f} last={rewards[-1]:.4f} "
                f"optimal={infos[-1].get('optimal', False)} kl={stats.get('kl', 0):.4f}"
            )
            if avg_r > best_avg + args.early_stop_delta:
                best_avg = avg_r
                patience_left = args.early_stop_patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    logger.info("[train] 触发提前停止 (early stopping)")
                    break

    # 10) 保存 LoRA 适配器、价值头与分词器，以及训练报告
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        if hasattr(policy, "save_pretrained"):
            policy.save_pretrained(out_dir / "lora_adapter")
            logger.info(f"已保存 LoRA 适配器到 {out_dir / 'lora_adapter'}")
    except Exception as e:
        logger.warning(f"保存 LoRA 适配器失败: {e}")
    try:
        if hasattr(policy, "v_head"):
            torch.save(policy.v_head.state_dict(), out_dir / "value_head.pt")
            logger.info(f"已保存价值头到 {out_dir / 'value_head.pt'}")
    except Exception as e:
        logger.warning(f"保存价值头失败: {e}")
    tok.save_pretrained(out_dir)
    logger.info(f"分词器已保存到 {out_dir}")

    final_recent = rewards_hist[-args.log_every:] if rewards_hist else []
    final_avg = (sum(final_recent) / max(1, len(final_recent))) if final_recent else 0.0
    report = {
        "episodes": step_idx + 1,
        "best_avg_reward": best_avg,
        "final_avg_reward": final_avg,
        "base_model": args.base_model,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
        "gen": {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature, "top_p": args.top_p},
    }
    with open(out_dir / "training_report.json", "w", encoding="utf-8") as wf:
        json.dump(report, wf, ensure_ascii=False, indent=2)
        logger.info(f"[train] 训练产物与报告已保存到 {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.path.join("data", "OR-Instruct-Data-3K", "OR-Instruct-Data-3K-Gurobi_sft.jsonl"))
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--sft_adapter_dir", default=None)
    ap.add_argument("--output_dir", default=os.path.join("data", "results", "QwenCoder7BSFT", "PPO-LORA"))
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--mini_batch_size", type=int, default=8)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=2)
    ap.add_argument("--ppo_epochs", type=int, default=4)
    ap.add_argument("--learning_rate", type=float, default=5e-6)
    ap.add_argument("--kl_coeff", type=float, default=0.8)
    ap.add_argument("--cliprange", type=float, default=0.1)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--exec_timeout", type=int, default=16)
    ap.add_argument("--reward_mode", default="tests")
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=128)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--bf16", action="store_true", default=True)
    # 参考模型4-bit量化开关（默认开启，可用 --no_ref_4bit 关闭）
    ap.add_argument("--ref_4bit", dest="ref_4bit", action="store_true", help="将参考模型以4-bit量化加载以降低显存")
    ap.add_argument("--no_ref_4bit", dest="ref_4bit", action="store_false")
    ap.set_defaults(ref_4bit=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--early_stop_delta", type=float, default=0.02)
    ap.add_argument("--early_stop_patience", type=int, default=5)
    ap.add_argument("--code_sim_weight", type=float, default=0.5, help="代码相似度奖励权重（最终奖励 += weight * similarity）")
    return ap


def main():
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()