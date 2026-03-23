import argparse
import json
import random
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_FILE = REPO_ROOT / "data" / "deduplicated_data.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.05
TEST_RATIO = 0.15


def parse_args():
    parser = argparse.ArgumentParser(
        description=f"按 {TRAIN_RATIO}:{VAL_RATIO}:{TEST_RATIO} 切分数据集"
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="输入数据文件路径",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="输出目录",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    return parser.parse_args()


def split_data(data, seed):
    # 先打乱数据，保证切分更均匀。
    shuffled_data = data[:]
    random.Random(seed).shuffle(shuffled_data)

    total = len(shuffled_data)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_data = shuffled_data[:train_end]
    val_data = shuffled_data[train_end:val_end]
    test_data = shuffled_data[val_end:]
    return train_data, val_data, test_data


def save_json(data, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_data, val_data, test_data = split_data(data, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_file = args.output_dir / "train.json"
    val_file = args.output_dir / "val.json"
    test_file = args.output_dir / "test.json"

    save_json(train_data, train_file)
    save_json(val_data, val_file)
    save_json(test_data, test_file)

    print(f"总条数: {len(data)}")
    print(f"训练集: {len(train_data)} -> {train_file}")
    print(f"验证集: {len(val_data)} -> {val_file}")
    print(f"测试集: {len(test_data)} -> {test_file}")
    print(f"随机种子: {args.seed}")


if __name__ == "__main__":
    main()
