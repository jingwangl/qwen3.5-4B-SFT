import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common.project_paths import build_data_path
from scripts.data_preprocess.common import load_dataset, sample_records, save_dataset

DEFAULT_TRAIN_FILE = build_data_path("train.json")
DEFAULT_VAL_FILE = build_data_path("val.json")
DEFAULT_OUTPUT_DIR = build_data_path("smoke")


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


def main():
    args = parse_args()

    train_data = load_dataset(args.train_file)
    val_data = load_dataset(args.val_file)

    smoke_train_data = sample_records(train_data, args.train_size, args.seed)
    smoke_val_data = sample_records(val_data, args.val_size, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    smoke_train_file = args.output_dir / "train.json"
    smoke_val_file = args.output_dir / "val.json"

    save_dataset(smoke_train_file, smoke_train_data)
    save_dataset(smoke_val_file, smoke_val_data)

    print(f"训练集抽样: {len(smoke_train_data)} -> {smoke_train_file}")
    print(f"验证集抽样: {len(smoke_val_data)} -> {smoke_val_file}")
    print(f"随机种子: {args.seed}")


if __name__ == "__main__":
    main()
