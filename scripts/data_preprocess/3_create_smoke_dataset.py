import argparse
import json
import random
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_FILE = REPO_ROOT / "data" / "train.json"
DEFAULT_VAL_FILE = REPO_ROOT / "data" / "val.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "smoke"


def parse_args():
    parser = argparse.ArgumentParser(description="生成 smoke test 数据")
    parser.add_argument(
        "--train-file",
        type=Path,
        default=DEFAULT_TRAIN_FILE,
        help="原始训练集路径",
    )
    parser.add_argument(
        "--val-file",
        type=Path,
        default=DEFAULT_VAL_FILE,
        help="原始验证集路径",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="输出目录",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=500,
        help="smoke 训练集条数",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=100,
        help="smoke 验证集条数",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    return parser.parse_args()


def load_json(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        return json.load(f)


def sample_data(data, size, seed):
    # 先打乱再截取，方便复现 smoke 数据。
    sampled_data = data[:]
    random.Random(seed).shuffle(sampled_data)
    return sampled_data[:size]


def save_json(data, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()

    train_data = load_json(args.train_file)
    val_data = load_json(args.val_file)

    smoke_train_data = sample_data(train_data, args.train_size, args.seed)
    smoke_val_data = sample_data(val_data, args.val_size, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    smoke_train_file = args.output_dir / "train.json"
    smoke_val_file = args.output_dir / "val.json"

    save_json(smoke_train_data, smoke_train_file)
    save_json(smoke_val_data, smoke_val_file)

    print(f"训练集抽样: {len(smoke_train_data)} -> {smoke_train_file}")
    print(f"验证集抽样: {len(smoke_val_data)} -> {smoke_val_file}")
    print(f"随机种子: {args.seed}")


if __name__ == "__main__":
    main()
