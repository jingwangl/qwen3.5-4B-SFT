import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common.project_paths import DATA_DIR, build_data_path
from scripts.data_preprocess.common import load_dataset, save_dataset, split_records_by_ratio

DEFAULT_INPUT_FILE = build_data_path("deduplicated_data.json")
DEFAULT_OUTPUT_DIR = DATA_DIR
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


def main():
    args = parse_args()

    data = load_dataset(args.input_file)
    train_data, val_data, test_data = split_records_by_ratio(
        data,
        seed=args.seed,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_file = args.output_dir / "train.json"
    val_file = args.output_dir / "val.json"
    test_file = args.output_dir / "test.json"

    save_dataset(train_file, train_data)
    save_dataset(val_file, val_data)
    save_dataset(test_file, test_data)

    print(f"总条数: {len(data)}")
    print(f"训练集: {len(train_data)} -> {train_file}")
    print(f"验证集: {len(val_data)} -> {val_file}")
    print(f"测试集: {len(test_data)} -> {test_file}")
    print(f"随机种子: {args.seed}")


if __name__ == "__main__":
    main()
