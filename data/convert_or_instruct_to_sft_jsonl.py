"""
Convert OR-Instruct-Data-3K-Gurobi.jsonl to SFT-ready JSONL with two-phase assistant output:
1) English mathematical modeling specification wrapped by MODEL_START/MODEL_END
2) A single executable Python code block (```python ... ```)

Usage:
  python training/convert_or_instruct_to_sft_jsonl.py \
    --input data/OR-Instruct-Data-3K/OR-Instruct-Data-3K-Gurobi.jsonl \
    --output data/OR-Instruct-Data-3K/OR-Instruct-Data-3K-Gurobi_sft.jsonl

Input record fields expected:
  - prompt: problem description (string)
  - completion: assistant response including modeling and a python code block (string)

Output JSONL per line:
  {
    "messages": [
      {"role": "system", "content": "You are an operations research assistant. First output a concise mathematical model specification (variables, objective, constraints) between MODEL_START and MODEL_END, then output a single executable Python code block. Do not output the final numeric result."},
      {"role": "user", "content": <prompt>},
      {"role": "assistant", "content": "MODEL_START\n<modeling_text>\nMODEL_END\n<python_code_block>"}
    ],
    "meta": {"source": "OR-Instruct-Data-3K-Gurobi"}
  }
"""

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(obj)
    return records


def _extract_first_python_block(text: str) -> Optional[str]:
    """Return the first ```python ... ``` block as a string (including fences)."""
    start = text.find("```python")
    if start == -1:
        return None
    end = text.find("```", start + len("```python"))
    if end == -1:
        return None
    inner = text[start + len("```python"): end]
    return f"```python\n{inner.strip()}\n```"


def _extract_modeling_text(text: str) -> str:
    """Extract modeling part before the first python code block and clean heading noise."""
    start = text.find("```python")
    modeling = text if start == -1 else text[:start]
    # Remove common heading like "## Python Code Solution ..." if present within modeling
    lines = [ln for ln in modeling.splitlines() if not ln.strip().lower().startswith("## python")]
    cleaned = "\n".join(lines).strip()
    return cleaned


def _ensure_code_block(code_str: str) -> str:
    """Ensure the code string is wrapped as a python code block."""
    if "```python" in code_str:
        return code_str
    return f"```python\n{code_str.strip()}\n```"


def convert_records(records: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for rec in records:
        prompt = rec.get("prompt") or rec.get("question") or rec.get("en_question") or ""
        completion = rec.get("completion") or rec.get("answer") or rec.get("response") or ""
        if not prompt or not completion:
            continue

        code_block = _extract_first_python_block(completion)
        if not code_block:
            # Skip entries without python code block
            continue
        modeling_text = _extract_modeling_text(completion)
        assistant_content = (
            "MODEL_START\n" + modeling_text + "\nMODEL_END\n" + _ensure_code_block(code_block)
        )

        yield {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an operations research assistant. First output a concise mathematical model "
                        "specification (variables, objective, constraints) between MODEL_START and MODEL_END, "
                        "then output a single executable Python code block. Do not output the final numeric result."
                    ),
                },
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant_content},
            ],
            "meta": {"source": "OR-Instruct-Data-3K-Gurobi"},
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default=os.path.join("data", "OR-Instruct-Data-3K", "OR-Instruct-Data-3K-Gurobi.jsonl"),
        help="Path to OR-Instruct-Data-3K-Gurobi JSONL",
    )
    ap.add_argument(
        "--output",
        default=os.path.join("data", "OR-Instruct-Data-3K", "OR-Instruct-Data-3K-Gurobi_sft.jsonl"),
        help="Path to output JSONL with messages",
    )
    args = ap.parse_args()

    records = _load_jsonl(args.input)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cnt_in, cnt_out = 0, 0
    with open(args.output, "w", encoding="utf-8") as wf:
        for obj in convert_records(records):
            cnt_in += 1
            wf.write(json.dumps(obj, ensure_ascii=False) + "\n")
            cnt_out += 1
    print(f"[convert] read={cnt_in} wrote={cnt_out} -> {args.output}")


if __name__ == "__main__":
    main()