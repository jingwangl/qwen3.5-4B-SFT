import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common.project_paths import build_data_path
from scripts.data_preprocess.common import deduplicate_records, load_dataset, save_dataset

INPUT_FILE = build_data_path("xlam_function_calling_60k.json")
OUTPUT_FILE = build_data_path("deduplicated_data.json")


def main():
    data = load_dataset(INPUT_FILE)

    deduplicated_data = deduplicate_records(data)
    save_dataset(OUTPUT_FILE, deduplicated_data)

    print(f"原始数据条数: {len(data)}")
    print(f"去重后条数: {len(deduplicated_data)}")
    print(f"删除重复条数: {len(data) - len(deduplicated_data)}")
    print(f"输出文件: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
