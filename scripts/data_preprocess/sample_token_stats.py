import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_FILE = REPO_ROOT / "data" / "train.json"
DEFAULT_MODEL_PATH = Path("/home/jingwangl/models/Qwen3.5-4B")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "stats"


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


def load_json(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        return json.load(f)


def build_sample_text(tokenizer, item):
    # 按完整训练样本来统计长度，结果更接近真实训练输入。
    messages = [
        {"role": "user", "content": item["query"]},
        {"role": "assistant", "tool_calls": json.loads(item["answers"])},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tools=json.loads(item["tools"]),
        tokenize=False,
        add_generation_prompt=False,
    )


def get_token_length(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


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


def save_json(data, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_output_stem(input_file):
    try:
        relative_path = input_file.resolve().relative_to(REPO_ROOT)
        return "_".join(relative_path.with_suffix("").parts)
    except ValueError:
        return input_file.stem


def main():
    args = parse_args()

    data = load_json(args.input_file)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    sample_lengths = []
    longest_item = None
    longest_length = -1

    for item in data:
        text = build_sample_text(tokenizer, item)
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

    stats = build_stats([item["token_length"] for item in sample_lengths])
    stats["input_file"] = str(args.input_file)
    stats["model_path"] = str(args.model_path)
    stats["longest_sample"] = longest_item

    args.output_dir.mkdir(parents=True, exist_ok=True)
    file_stem = build_output_stem(args.input_file)
    lengths_file = args.output_dir / f"{file_stem}_token_lengths.json"
    summary_file = args.output_dir / f"{file_stem}_token_summary.json"

    save_json(sample_lengths, lengths_file)
    save_json(stats, summary_file)

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
