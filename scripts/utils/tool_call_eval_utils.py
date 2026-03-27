import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

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


PreparedEvalItem = tuple[EvalExample, str, int]


@dataclass
class EvalBatchPlan:
    items: list[PreparedEvalItem]
    prompt_token_count: int
    max_prompt_length: int
    estimated_token_count: int
    exceeds_token_budget: bool = False


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
    import torch

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
    valid_json, json_calls = parse_json_tool_calls(text)
    if valid_json:
        return json_calls

    calls: list[dict[str, Any]] = []
    for function_name, body in TOOL_CALL_PATTERN.findall(text):
        arguments: dict[str, Any] = {}
        for param_name, raw_value in PARAM_PATTERN.findall(body):
            arguments[param_name.strip()] = maybe_decode_json_scalar(raw_value)
        calls.append({"name": function_name.strip(), "arguments": arguments})
    return calls


def parse_json_tool_calls(text: str) -> tuple[bool, list[dict[str, Any]]]:
    stripped = text.strip()
    if not stripped:
        return False, []

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return False, []

    return True, normalize_json_tool_calls(payload)


def normalize_json_tool_calls(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        if "tool_calls" in payload:
            tool_calls = payload.get("tool_calls")
            if isinstance(tool_calls, list):
                return [
                    normalized
                    for item in tool_calls
                    if (normalized := normalize_json_tool_call(item)) is not None
                ]
            return []

        normalized = normalize_json_tool_call(payload)
        return [normalized] if normalized is not None else []

    if isinstance(payload, list):
        return [
            normalized
            for item in payload
            if (normalized := normalize_json_tool_call(item)) is not None
        ]

    return []


def normalize_json_tool_call(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None

    if "name" in payload:
        name = payload.get("name")
        arguments = payload.get("arguments", {})
    elif isinstance(payload.get("function"), dict):
        function_payload = payload["function"]
        name = function_payload.get("name")
        arguments = function_payload.get("arguments", {})
    else:
        return None

    if not isinstance(name, str) or not name.strip():
        return None

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            pass

    return {
        "name": name.strip(),
        "arguments": arguments,
    }


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


def serialize_call(call: dict[str, Any]) -> str:
    return json.dumps(call, ensure_ascii=False, sort_keys=True)


def calls_equal_ordered(predicted: list[dict[str, Any]], gold: list[dict[str, Any]]) -> bool:
    pred_norm = [canonicalize_call(call) for call in predicted]
    gold_norm = [canonicalize_call(call) for call in gold]
    return pred_norm == gold_norm


def calls_equal_unordered(predicted: list[dict[str, Any]], gold: list[dict[str, Any]]) -> bool:
    pred_norm = sorted(serialize_call(canonicalize_call(call)) for call in predicted)
    gold_norm = sorted(serialize_call(canonicalize_call(call)) for call in gold)
    return pred_norm == gold_norm


def names_equal_unordered(predicted: list[dict[str, Any]], gold: list[dict[str, Any]]) -> bool:
    pred_names = sorted(call["name"] for call in predicted)
    gold_names = sorted(call["name"] for call in gold)
    return pred_names == gold_names


def extract_unparsed_text(text: str) -> str:
    fragments: list[str] = []
    cursor = 0
    for match in TOOL_CALL_PATTERN.finditer(text):
        fragments.append(text[cursor : match.start()])
        cursor = match.end()
    fragments.append(text[cursor:])
    return "".join(fragments).strip()


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def multiset_difference(left_items: list[str], right_items: list[str]) -> list[str]:
    diff_counter = Counter(left_items) - Counter(right_items)
    items: list[str] = []
    for name in sorted(diff_counter):
        items.extend([name] * diff_counter[name])
    return items


def evaluate_tool_call_prediction(
    predicted_calls: list[dict[str, Any]],
    gold_calls: list[dict[str, Any]],
    generated_text: str,
) -> dict[str, Any]:
    valid_json_object, json_predicted_calls = parse_json_tool_calls(generated_text)
    predicted_norm = [canonicalize_call(call) for call in predicted_calls]
    json_predicted_norm = [canonicalize_call(call) for call in json_predicted_calls]
    gold_norm = [canonicalize_call(call) for call in gold_calls]

    predicted_names = [call["name"] for call in predicted_norm]
    gold_names = [call["name"] for call in gold_norm]
    predicted_call_strings = [serialize_call(call) for call in predicted_norm]
    gold_call_strings = [serialize_call(call) for call in gold_norm]
    tool_selection_correct = sorted(predicted_names) == sorted(gold_names)
    call_correct = sorted(predicted_call_strings) == sorted(gold_call_strings)
    return {
        "valid_json_object": valid_json_object,
        "tool_selection_correct": tool_selection_correct,
        "call_correct": call_correct,
        "predicted_calls": predicted_norm,
        "json_predicted_calls": json_predicted_norm,
        "missing_tool_names": multiset_difference(gold_names, predicted_names),
        "extra_tool_names": multiset_difference(predicted_names, gold_names),
    }


def build_generation_prompt(tokenizer: Any, example: EvalExample) -> str:
    messages = [{"role": "user", "content": example.query}]
    return tokenizer.apply_chat_template(
        messages,
        tools=example.tools,
        tokenize=False,
        add_generation_prompt=True,
    )


def prepare_examples_for_batching(
    tokenizer: Any,
    examples: list[EvalExample],
    bucket_by_length: bool,
) -> list[PreparedEvalItem]:
    prompts = [build_generation_prompt(tokenizer, example) for example in examples]
    tokenized = tokenizer(
        prompts,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    prompt_lengths = [len(ids) for ids in tokenized["input_ids"]]

    prepared = list(zip(examples, prompts, prompt_lengths))
    if bucket_by_length:
        prepared.sort(key=lambda item: item[2], reverse=True)
    return prepared


def build_dynamic_batches(
    prepared_examples: list[PreparedEvalItem],
    first_batch_size: int,
    max_batch_size: int,
    max_new_tokens: int,
) -> tuple[list[EvalBatchPlan], int]:
    if first_batch_size <= 0:
        raise ValueError("first_batch_size 必须大于 0。")
    if max_batch_size <= 0:
        raise ValueError("max_batch_size 必须大于 0。")
    if max_batch_size < first_batch_size:
        raise ValueError("max_batch_size 不能小于 first_batch_size。")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens 必须大于 0。")
    if not prepared_examples:
        return [], 0

    initial_count = min(first_batch_size, len(prepared_examples))
    initial_items = prepared_examples[:initial_count]
    initial_prompt_token_count = sum(item[2] for item in initial_items)
    initial_max_prompt_length = max(item[2] for item in initial_items)
    token_budget = initial_count * (initial_max_prompt_length + max_new_tokens)
    batches = [
        EvalBatchPlan(
            items=list(initial_items),
            prompt_token_count=initial_prompt_token_count,
            max_prompt_length=initial_max_prompt_length,
            estimated_token_count=token_budget,
        )
    ]

    current_items: list[PreparedEvalItem] = []
    current_token_count = 0
    current_max_prompt_length = 0
    for item in prepared_examples[initial_count:]:
        prompt_length = item[2]
        single_item_estimated_token_count = prompt_length + max_new_tokens
        if single_item_estimated_token_count > token_budget:
            if current_items:
                batches.append(
                    EvalBatchPlan(
                        items=current_items,
                        prompt_token_count=current_token_count,
                        max_prompt_length=current_max_prompt_length,
                        estimated_token_count=len(current_items)
                        * (current_max_prompt_length + max_new_tokens),
                    )
                )
                current_items = []
                current_token_count = 0
                current_max_prompt_length = 0
            batches.append(
                EvalBatchPlan(
                    items=[item],
                    prompt_token_count=prompt_length,
                    max_prompt_length=prompt_length,
                    estimated_token_count=single_item_estimated_token_count,
                    exceeds_token_budget=True,
                )
            )
            continue

        next_batch_size = len(current_items) + 1
        next_token_count = current_token_count + prompt_length
        next_max_prompt_length = max(current_max_prompt_length, prompt_length)
        next_estimated_token_count = next_batch_size * (next_max_prompt_length + max_new_tokens)
        if current_items and (
            next_batch_size > max_batch_size or next_estimated_token_count > token_budget
        ):
            batches.append(
                EvalBatchPlan(
                    items=current_items,
                    prompt_token_count=current_token_count,
                    max_prompt_length=current_max_prompt_length,
                    estimated_token_count=len(current_items)
                    * (current_max_prompt_length + max_new_tokens),
                )
            )
            current_items = [item]
            current_token_count = prompt_length
            current_max_prompt_length = prompt_length
            continue

        current_items.append(item)
        current_token_count = next_token_count
        current_max_prompt_length = next_max_prompt_length

    if current_items:
        batches.append(
            EvalBatchPlan(
                items=current_items,
                prompt_token_count=current_token_count,
                max_prompt_length=current_max_prompt_length,
                estimated_token_count=len(current_items) * (current_max_prompt_length + max_new_tokens),
            )
        )

    return batches, token_budget


def generate_batch(
    tokenizer: Any,
    model: Any,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
) -> tuple[list[tuple[str, list[dict[str, Any]]]], int]:
    import torch

    model_device = next(model.parameters()).device
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model_device)

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

    outputs: list[tuple[str, list[dict[str, Any]]]] = []
    input_lengths = inputs["attention_mask"].sum(dim=1)
    generated_token_count = 0
    for row_index in range(output_ids.shape[0]):
        generated_ids = output_ids[row_index][int(input_lengths[row_index]) :]
        generated_token_count += int(generated_ids.shape[0])
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        outputs.append((generated_text, parse_tool_calls(generated_text)))
    return outputs, generated_token_count


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    tool_selection_correct_count = sum(item["tool_selection_correct"] for item in results)
    call_correct_count = sum(item["call_correct"] for item in results)

    return {
        "num_examples": total,
        "tool_selection_accuracy": safe_divide(tool_selection_correct_count, total),
        "call_accuracy": safe_divide(call_correct_count, total),
    }
