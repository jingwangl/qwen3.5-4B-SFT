import sys
from typing import Any, Protocol

from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.utils.common import load_json_file, save_json_file
from scripts.utils.tool_call_eval_utils import (
    calls_equal_ordered,
    calls_equal_unordered,
    choose_examples,
    generate_one,
    get_torch_dtype,
    names_equal_unordered,
    summarize_results,
)


class ToolCallEvalConfigLike(Protocol):
    base_model_path: str
    adapter_path: Any
    data_path: Any
    output_dir: Any
    num_samples: int
    seed: int
    max_new_tokens: int
    temperature: float
    sample_mode: str
    dtype: str
    tokenizer_path: str | Any
    model_path: str
    model_type: str

    def build_summary_path(self, total_rows: int): ...
    def build_details_path(self, total_rows: int): ...
    def build_summary_metadata(self, total_rows: int) -> dict: ...


def print_run_info(config: ToolCallEvalConfigLike, total_rows: int, total_examples: int) -> None:
    print(f"Loaded {total_rows} rows from {config.data_path}")
    print(f"Evaluating {total_examples} samples with sample_mode={config.sample_mode}")
    print(f"Model type: {config.model_type}")
    print(f"Model path: {config.model_path}")
    if config.adapter_path is not None:
        print(f"Base model path: {config.base_model_path}")


def load_model_and_tokenizer(config: ToolCallEvalConfigLike):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_path,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_path,
            torch_dtype=get_torch_dtype(config.dtype),
            device_map="auto",
            trust_remote_code=True,
        )
        model = base_model
        if config.adapter_path is not None:
            model = PeftModel.from_pretrained(base_model, config.adapter_path)

        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None and generation_config.pad_token_id is None:
            generation_config.pad_token_id = tokenizer.pad_token_id
        model.eval()
    except Exception as exc:
        print("\nModel load failed.\n", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        if config.adapter_path is not None:
            print(
                f"\nCheck both the base model path and adapter path:\n"
                f"base_model_path={config.base_model_path}\n"
                f"adapter_path={config.adapter_path}",
                file=sys.stderr,
            )
        else:
            print(
                "\nThis usually means the installed Transformers build does not yet "
                "support the checkpoint architecture. Verify that your environment "
                "supports model_type=qwen3_5 before running this evaluator.",
                file=sys.stderr,
            )
        return None, None

    return model, tokenizer


def evaluate_examples(config: ToolCallEvalConfigLike, examples, model, tokenizer) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    # 这里保留最核心的逐条评估逻辑，方便后面继续扩展指标。
    progress_bar = tqdm(examples, desc="评测中")
    for index, example in enumerate(progress_bar, start=1):
        generated_text, predicted_calls = generate_one(
            tokenizer=tokenizer,
            model=model,
            example=example,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
        )
        gold_calls = example.gold_calls
        record = {
            "dataset_index": example.index,
            "query": example.query,
            "tools": example.tools,
            "gold_calls": gold_calls,
            "predicted_text": generated_text,
            "predicted_calls": predicted_calls,
            "ordered_exact_match": calls_equal_ordered(predicted_calls, gold_calls),
            "unordered_exact_match": calls_equal_unordered(predicted_calls, gold_calls),
            "unordered_name_match": names_equal_unordered(predicted_calls, gold_calls),
            "tool_count_match": len(predicted_calls) == len(gold_calls),
        }
        results.append(record)
        progress_bar.set_postfix(
            sample=f"{index}/{len(examples)}",
            idx=example.index,
            ordered=record["ordered_exact_match"],
            unordered=record["unordered_exact_match"],
            name=record["unordered_name_match"],
        )

    return results


def save_results(
    config: ToolCallEvalConfigLike,
    total_rows: int,
    summary: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    save_json_file(config.build_summary_path(total_rows), summary)
    save_json_file(config.build_details_path(total_rows), results)


def run_tool_call_eval(config: ToolCallEvalConfigLike) -> int:
    rows = load_json_file(config.data_path)
    examples = choose_examples(rows, config.num_samples, config.seed, config.sample_mode)
    total_rows = len(rows)
    print_run_info(config, total_rows, len(examples))

    model, tokenizer = load_model_and_tokenizer(config)
    if model is None or tokenizer is None:
        return 1

    results = evaluate_examples(config, examples, model, tokenizer)
    summary = config.build_summary_metadata(total_rows)
    summary.update(summarize_results(results))
    save_results(config, total_rows, summary, results)

    print("\n=== Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"\nSaved summary to {config.build_summary_path(total_rows)}")
    print(f"Saved details to {config.build_details_path(total_rows)}")
    return 0
