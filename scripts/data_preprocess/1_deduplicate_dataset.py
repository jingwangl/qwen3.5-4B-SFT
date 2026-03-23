import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = REPO_ROOT / "data" / "xlam_function_calling_60k.json"
OUTPUT_FILE = REPO_ROOT / "data" / "deduplicated_data.json"


def normalize_json_string(text):
    # tools 和 answers 本身是 JSON 字符串，这里转成统一格式再比较。
    return json.dumps(json.loads(text), ensure_ascii=False, sort_keys=True)


def deduplicate_data(data):
    seen = set()
    result = []

    for item in data:
        key = (
            item["query"].strip(),
            normalize_json_string(item["tools"]),
            normalize_json_string(item["answers"]),
        )
        if key in seen:
            continue

        seen.add(key)
        result.append(item)

    return result


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    deduplicated_data = deduplicate_data(data)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(deduplicated_data, f, ensure_ascii=False, indent=2)

    print(f"原始数据条数: {len(data)}")
    print(f"去重后条数: {len(deduplicated_data)}")
    print(f"删除重复条数: {len(data) - len(deduplicated_data)}")
    print(f"输出文件: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
