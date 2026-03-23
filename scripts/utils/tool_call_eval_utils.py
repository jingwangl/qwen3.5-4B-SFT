import json
import random
import re
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.utils.common import load_json_value


TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*<function=([^>\n]+)>\s*(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
PARAM_PATTERN = re.compile(
    r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)


@dataclass
class EvalExample:
    index: int
    query: str
    tools: list[dict[str, Any]]
    gold_calls: list[dict[str, Any]]


def choose_examples(
    rows: list[dict[str, Any]],
    num_samples: int,
    seed: int,
    sample_mode: str,
) -> list[EvalExample]:
    total = len(rows) if num_samples == -1 else min(num_samples, len(rows))
    indices = list(range(len(rows)))
    if sample_mode == "random":
        rng = random.Random(seed)
        indices = rng.sample(indices, total)
    else:
        indices = indices[:total]

    examples: list[EvalExample] = []
    for idx in indices:
        row = rows[idx]
        examples.append(
            EvalExample(
                index=idx,
                query=row["query"],
                tools=load_json_value(row["tools"]),
                gold_calls=load_json_value(row["answers"]),
            )
        )
    return examples


def get_torch_dtype(name: str) -> Any:
    if name == "auto":
        return "auto"
    return getattr(torch, name)


def maybe_decode_json_scalar(raw: str) -> Any:
    text = raw.strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None

    if re.fullmatch(r"-?\d+", text):
        try:
            return int(text)
        except ValueError:
            return text
    if re.fullmatch(r"-?\d+\.\d+", text):
        try:
            return float(text)
        except ValueError:
            return text
    return text


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for function_name, body in TOOL_CALL_PATTERN.findall(text):
        arguments: dict[str, Any] = {}
        for param_name, raw_value in PARAM_PATTERN.findall(body):
            arguments[param_name.strip()] = maybe_decode_json_scalar(raw_value)
        calls.append({"name": function_name.strip(), "arguments": arguments})
    return calls


def canonicalize_scalar(value: Any) -> Any:
    if isinstance(value, str):
        compact = " ".join(value.split())
        lowered = compact.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered == "null":
            return None
        if re.fullmatch(r"-?\d+", compact):
            try:
                return int(compact)
            except ValueError:
                return compact
        if re.fullmatch(r"-?\d+\.\d+", compact):
            try:
                return float(compact)
            except ValueError:
                return compact
        return compact
    return value


def canonicalize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: canonicalize_json(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [canonicalize_json(item) for item in value]
    return canonicalize_scalar(value)


def canonicalize_call(call: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": call["name"],
        "arguments": canonicalize_json(call.get("arguments", {})),
    }


def calls_equal_ordered(predicted: list[dict[str, Any]], gold: list[dict[str, Any]]) -> bool:
    pred_norm = [canonicalize_call(call) for call in predicted]
    gold_norm = [canonicalize_call(call) for call in gold]
    return pred_norm == gold_norm


def calls_equal_unordered(predicted: list[dict[str, Any]], gold: list[dict[str, Any]]) -> bool:
    pred_norm = sorted(
        json.dumps(canonicalize_call(call), ensure_ascii=False, sort_keys=True)
        for call in predicted
    )
    gold_norm = sorted(
        json.dumps(canonicalize_call(call), ensure_ascii=False, sort_keys=True)
        for call in gold
    )
    return pred_norm == gold_norm


def names_equal_unordered(predicted: list[dict[str, Any]], gold: list[dict[str, Any]]) -> bool:
    pred_names = sorted(call["name"] for call in predicted)
    gold_names = sorted(call["name"] for call in gold)
    return pred_names == gold_names


def generate_one(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    example: EvalExample,
    max_new_tokens: int,
    temperature: float,
) -> tuple[str, list[dict[str, Any]]]:
    messages = [{"role": "user", "content": example.query}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tools=example.tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
    }
    # 显式传 pad_token_id，避免 generate 反复打印提示。
    if tokenizer.pad_token_id is not None:
        generate_kwargs["pad_token_id"] = tokenizer.pad_token_id
    elif tokenizer.eos_token_id is not None:
        generate_kwargs["pad_token_id"] = tokenizer.eos_token_id
    if temperature > 0:
        generate_kwargs["temperature"] = temperature

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generate_kwargs)

    generated_ids = output_ids[0][inputs.input_ids.shape[1] :]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text, parse_tool_calls(generated_text)


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    ordered_exact = sum(item["ordered_exact_match"] for item in results)
    unordered_exact = sum(item["unordered_exact_match"] for item in results)
    unordered_name = sum(item["unordered_name_match"] for item in results)
    count_match = sum(item["tool_count_match"] for item in results)
    parsed_any = sum(bool(item["predicted_calls"]) for item in results)
    return {
        "num_examples": total,
        "ordered_exact_match": ordered_exact / total if total else 0.0,
        "unordered_exact_match": unordered_exact / total if total else 0.0,
        "unordered_name_match": unordered_name / total if total else 0.0,
        "tool_count_match": count_match / total if total else 0.0,
        "produced_any_tool_call": parsed_any / total if total else 0.0,
    }
