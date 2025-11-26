"""
Convert IndustryOR augmented data to SFT-ready JSONL with two-phase assistant output:
1) English mathematical modeling specification wrapped by MODEL_START/MODEL_END
2) A single executable Python code block (```python ... ```)

Usage:
  python training/convert_industryor_to_sft_jsonl.py \
    --input data/augment_data/IndustryOR/IndustryOR.json \
    --output data/augment_data/IndustryOR/IndustryOR_sft.jsonl

Input record fields expected:
  - en_question: problem description (string)
  - math_model: modeling explanation in English (string)
  - gurobi_code: Python code as markdown code block (string)
  - en_answer: expected numeric result (optional, not used in text)

Output JSONL per line:
  {
    "messages": [
      {"role": "system", "content": "You are an operations research assistant. First output a concise mathematical model specification (variables, objective, constraints) between MODEL_START and MODEL_END, then output a single executable Python code block. Do not output the final numeric result."},
      {"role": "user", "content": <en_question>},
      {"role": "assistant", "content": "MODEL_START\n<math_model>\nMODEL_END\n<gurobi_code>"}
    ],
    "meta": {"source": "IndustryOR", "answer": <en_answer|optional>}
  }
"""

import argparse
import json
import os
from typing import Any, Dict, Iterable, List


def _load_json_any(input_path: str) -> List[Dict[str, Any]]:
    """Load JSON file that may be a list of objects or a single object or JSONL.

    Returns a list of dicts.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Try standard JSON load first
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Common patterns: {"data": [...]} or a single record
            if "data" in data and isinstance(data["data"], list):
                return data["data"]
            return [data]
    except json.JSONDecodeError:
        # Fallback: treat as JSONL
        records: List[Dict[str, Any]] = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    records.append(obj)
                except json.JSONDecodeError:
                    # Skip malformed lines
                    continue
        if records:
            return records
        raise


def _ensure_code_block(code_str: str) -> str:
    """Ensure the code string is wrapped as a python code block.

    If it already contains ```python, return as-is; otherwise wrap.
    """
    if "```python" in code_str:
        return code_str
    # Strip trailing backticks if any accidental formatting
    cleaned = code_str.strip()
    return f"```python\n{cleaned}\n```"


def convert_records(records: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for rec in records:
        en_question = rec.get("en_question") or rec.get("question") or ""
        math_model = rec.get("math_model") or rec.get("modeling") or ""
        gurobi_code = rec.get("gurobi_code") or rec.get("code") or ""
        en_answer = rec.get("en_answer") or rec.get("answer")

        if not en_question or not math_model or not gurobi_code:
            # skip incomplete records
            continue

        assistant_content = (
            "MODEL_START\n" + math_model.strip() + "\nMODEL_END\n" + _ensure_code_block(gurobi_code)
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
                {"role": "user", "content": en_question},
                {"role": "assistant", "content": assistant_content},
            ],
            "meta": {"source": "IndustryOR", **({"answer": en_answer} if en_answer is not None else {})},
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default=os.path.join("data", "augment_data", "IndustryOR", "IndustryOR.json"),
        help="Path to IndustryOR augmented JSON",
    )
    ap.add_argument(
        "--output",
        default=os.path.join("data", "augment_data", "IndustryOR", "IndustryOR_sft.jsonl"),
        help="Path to output JSONL with messages",
    )
    args = ap.parse_args()

    records = _load_json_any(args.input)
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