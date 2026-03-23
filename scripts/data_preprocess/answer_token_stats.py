import argparse
import os
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utils.common import load_json_file, save_json_file  # noqa: E402
from scripts.utils.train_utils import build_full_text, build_prompt_text  # noqa: E402


DEFAULT_INPUT_FILE = REPO_ROOT / "data" / "xlam_function_calling_60k.json"
DEFAULT_MODEL_PATH = Path(
    os.environ.get(
        "QWEN_MODEL_PATH",
        REPO_ROOT.parent / "models" / "Qwen3.5-4B",
    )
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "stats"


def parse_args():
    parser = argparse.ArgumentParser(description="统计数据集 answer 的 token 长度")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="输入数据文件路径",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="tokenizer 路径",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="统计结果输出目录",
    )
    return parser.parse_args()


def get_answer_token_length(tokenizer, item):
    # 用完整对话长度减去 prompt 长度，更接近评测时需要的 max_new_tokens。
    prompt_text = build_prompt_text(tokenizer, item)
    full_text = build_full_text(tokenizer, item)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    return max(0, len(full_ids) - len(prompt_ids))


def build_stats(lengths):
    length_array = np.array(lengths)
    return {
        "num_samples": int(length_array.size),
        "p50": float(np.percentile(length_array, 50)),
        "p90": float(np.percentile(length_array, 90)),
        "p95": float(np.percentile(length_array, 95)),
        "p99": float(np.percentile(length_array, 99)),
        "max": int(length_array.max()),
    }


def build_output_stem(input_file):
    try:
        relative_path = input_file.resolve().relative_to(REPO_ROOT)
        return "_".join(relative_path.with_suffix("").parts)
    except ValueError:
        return input_file.stem


def main():
    args = parse_args()

    data = load_json_file(args.input_file)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    answer_lengths = []
    longest_item = None
    longest_length = -1

    for item in data:
        token_length = get_answer_token_length(tokenizer, item)
        answer_lengths.append(
            {
                "id": item.get("id"),
                "token_length": token_length,
            }
        )

        if token_length > longest_length:
            longest_length = token_length
            longest_item = {
                "id": item.get("id"),
                "query": item["query"],
                "answers": item["answers"],
                "token_length": token_length,
            }

    stats = build_stats([item["token_length"] for item in answer_lengths])
    stats["input_file"] = str(args.input_file)
    stats["model_path"] = str(args.model_path)
    stats["longest_answer"] = longest_item
    stats["recommended_max_new_tokens"] = int(stats["max"] + 32)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    file_stem = build_output_stem(args.input_file)
    lengths_file = args.output_dir / f"{file_stem}_answer_token_lengths.json"
    summary_file = args.output_dir / f"{file_stem}_answer_token_summary.json"

    save_json_file(lengths_file, answer_lengths)
    save_json_file(summary_file, stats)

    print(f"样本数: {stats['num_samples']}")
    print(f"P50: {stats['p50']:.2f}")
    print(f"P90: {stats['p90']:.2f}")
    print(f"P95: {stats['p95']:.2f}")
    print(f"P99: {stats['p99']:.2f}")
    print(f"最长 answer token 长度: {stats['max']}")
    print(f"建议 max_new_tokens: {stats['recommended_max_new_tokens']}")
    print(f"最长样本 id: {longest_item['id']}")
    print(f"每条 answer 长度: {lengths_file}")
    print(f"汇总结果: {summary_file}")


if __name__ == "__main__":
    main()
