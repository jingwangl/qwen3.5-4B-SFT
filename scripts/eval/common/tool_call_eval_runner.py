import sys
import time
from typing import Any, Protocol

from peft import PeftModel
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.utils.common import load_json_file, save_json_file
from scripts.utils.tool_call_eval_utils import (
    build_dynamic_batches,
    choose_examples,
    evaluate_tool_call_prediction,
    generate_batch,
    get_torch_dtype,
    prepare_examples_for_batching,
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
    batch_size: int
    max_batch_size: int
    bucket_by_length: bool
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
    print(f"First batch size: {config.batch_size}")
    print(f"Max batch size: {config.max_batch_size}")
    print(f"Bucket by length: {config.bucket_by_length}")
    print(f"Model type: {config.model_type}")
    print(f"Model path: {config.model_path}")
    if config.adapter_path is not None:
        print(f"Base model path: {config.base_model_path}")


def format_cuda_mem_gib(memory_bytes: int) -> str:
    return f"{memory_bytes / (1024**3):.2f} GiB"


def format_cuda_memory_log_line(tokens_per_sec: float) -> str | None:
    if not torch.cuda.is_available():
        return None
    device_index = torch.cuda.current_device()
    total_mem = torch.cuda.get_device_properties(device_index).total_memory

    allocated = torch.cuda.memory_allocated(device_index)
    reserved = torch.cuda.memory_reserved(device_index)
    peak_alloc = torch.cuda.max_memory_allocated(device_index)
    peak_reserved = torch.cuda.max_memory_reserved(device_index)

    alloc_ratio = (allocated / total_mem) * 100
    reserved_ratio = (reserved / total_mem) * 100
    peak_alloc_ratio = (peak_alloc / total_mem) * 100
    peak_reserved_ratio = (peak_reserved / total_mem) * 100

    return (
        f"| tok/s={tokens_per_sec:.2f} "
        f"| alloc={format_cuda_mem_gib(allocated)} ({alloc_ratio:.1f}%) "
        f"| reserved={format_cuda_mem_gib(reserved)} ({reserved_ratio:.1f}%) "
        f"| peak_alloc={format_cuda_mem_gib(peak_alloc)} ({peak_alloc_ratio:.1f}%) "
        f"| peak_reserved={format_cuda_mem_gib(peak_reserved)} ({peak_reserved_ratio:.1f}%)"
    )


def load_model_and_tokenizer(config: ToolCallEvalConfigLike):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_path,
            trust_remote_code=True,
        )
        tokenizer.padding_side = "left"
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


def evaluate_examples(
    config: ToolCallEvalConfigLike,
    examples,
    model,
    tokenizer,
) -> tuple[list[dict[str, Any]], int]:
    results: list[dict[str, Any]] = []
    prepared_examples = prepare_examples_for_batching(
        tokenizer=tokenizer,
        examples=examples,
        bucket_by_length=config.bucket_by_length,
    )
    batch_plans, dynamic_batch_token_budget = build_dynamic_batches(
        prepared_examples=prepared_examples,
        first_batch_size=config.batch_size,
        max_batch_size=config.max_batch_size,
    )
    print(f"Dynamic batch token budget: {dynamic_batch_token_budget}")
    print(f"Prepared {len(batch_plans)} batches")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    progress_bar = tqdm(total=len(examples), desc="评测中")
    for batch_index, batch_plan in enumerate(batch_plans, start=1):
        batch_items = batch_plan.items
        batch_examples = [item[0] for item in batch_items]
        batch_prompts = [item[1] for item in batch_items]
        print(
            f"Batch {batch_index}/{len(batch_plans)}: "
            f"size={len(batch_items)} prompt_tokens={batch_plan.prompt_token_count}"
        )
        if batch_plan.exceeds_token_budget and batch_examples:
            print(
                "Warning: sample "
                f"{batch_examples[0].index} prompt_tokens={batch_plan.prompt_token_count} "
                f"exceeds dynamic budget={dynamic_batch_token_budget}; running it alone."
            )
        batch_begin = time.perf_counter()
        batch_outputs, generated_token_count = generate_batch(
            tokenizer=tokenizer,
            model=model,
            prompts=batch_prompts,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
        )
        batch_duration = max(time.perf_counter() - batch_begin, 1e-8)
        tokens_per_sec = generated_token_count / batch_duration

        for example, (generated_text, predicted_calls) in zip(batch_examples, batch_outputs):
            gold_calls = example.gold_calls
            metrics = evaluate_tool_call_prediction(predicted_calls, gold_calls, generated_text)
            record = {
                "dataset_index": example.index,
                "query": example.query,
                "tools": example.tools,
                "gold_calls": gold_calls,
                "predicted_text": generated_text,
                "predicted_calls": predicted_calls,
                **metrics,
            }
            results.append(record)
            progress_bar.update(1)
            progress_bar.set_postfix(
                sample=f"{len(results)}/{len(examples)}",
                idx=example.index,
                json=record["valid_json_object"],
                tool=record["tool_selection_correct"],
                call=record["call_correct"],
            )

        cuda_log_line = format_cuda_memory_log_line(tokens_per_sec)
        if cuda_log_line is not None:
            print(cuda_log_line)

    progress_bar.close()

    return results, dynamic_batch_token_budget


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

    results, dynamic_batch_token_budget = evaluate_examples(config, examples, model, tokenizer)
    summary = config.build_summary_metadata(total_rows)
    summary["dynamic_batch_token_budget"] = dynamic_batch_token_budget
    summary.update(summarize_results(results))
    save_results(config, total_rows, summary, results)

    print("\n=== Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"\nSaved summary to {config.build_summary_path(total_rows)}")
    print(f"Saved details to {config.build_details_path(total_rows)}")
    return 0
