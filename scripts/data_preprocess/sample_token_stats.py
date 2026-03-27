import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common.project_paths import build_data_path, get_default_model_path  # noqa: E402
from scripts.common.tool_call_dataset import build_full_text, extract_tool_call_sample_fields  # noqa: E402
from scripts.data_preprocess.common import (  # noqa: E402
    build_length_stats,
    build_output_stem,
    load_dataset,
    save_dataset,
)


DEFAULT_INPUT_FILE = build_data_path("train.json")
DEFAULT_MODEL_PATH = Path(get_default_model_path())
DEFAULT_OUTPUT_DIR = build_data_path("stats")


def parse_args():
    parser = argparse.ArgumentParser(description="统计样本 token 长度")
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


def get_token_length(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def main():
    args = parse_args()

    data = load_dataset(args.input_file)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    sample_lengths = []
    longest_item = None
    longest_length = -1

    for item in data:
        query, tools, answers = extract_tool_call_sample_fields(item)
        text = build_full_text(tokenizer, query, tools, answers)
        token_length = get_token_length(tokenizer, text)

        sample_lengths.append(
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
                "token_length": token_length,
            }

    stats = build_length_stats([item["token_length"] for item in sample_lengths])
    stats["input_file"] = str(args.input_file)
    stats["model_path"] = str(args.model_path)
    stats["longest_sample"] = longest_item

    args.output_dir.mkdir(parents=True, exist_ok=True)
    file_stem = build_output_stem(args.input_file)
    lengths_file = args.output_dir / f"{file_stem}_token_lengths.json"
    summary_file = args.output_dir / f"{file_stem}_token_summary.json"

    save_dataset(lengths_file, sample_lengths)
    save_dataset(summary_file, stats)

    print(f"样本数: {stats['num_samples']}")
    print(f"P50: {stats['p50']:.2f}")
    print(f"P90: {stats['p90']:.2f}")
    print(f"P95: {stats['p95']:.2f}")
    print(f"P99: {stats['p99']:.2f}")
    print(f"最长样本: {stats['max']}")
    print(f"最长样本 id: {longest_item['id']}")
    print(f"每条样本长度: {lengths_file}")
    print(f"汇总结果: {summary_file}")


if __name__ == "__main__":
    main()
