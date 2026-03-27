import json
import math
import random
from pathlib import Path
from typing import Any, Sequence

from scripts.common.project_paths import REPO_ROOT
from scripts.utils.common import load_json_file, save_json_file


def normalize_json_string(text: str) -> str:
    # tools 和 answers 本身是 JSON 字符串，这里转成统一格式再比较。
    return json.dumps(json.loads(text), ensure_ascii=False, sort_keys=True)


def deduplicate_records(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    result: list[dict[str, Any]] = []

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


def shuffle_copy(data: Sequence[Any], seed: int) -> list[Any]:
    shuffled = list(data)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def sample_records(data: Sequence[Any], size: int, seed: int) -> list[Any]:
    return shuffle_copy(data, seed)[:size]


def split_records_by_ratio(
    data: Sequence[Any],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[Any], list[Any], list[Any]]:
    shuffled = shuffle_copy(data, seed)
    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def percentile(values: Sequence[int], percent: float) -> float:
    if not values:
        raise ValueError("values 不能为空。")

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    rank = (percent / 100.0) * (len(sorted_values) - 1)
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    if lower_index == upper_index:
        return float(lower_value)
    fraction = rank - lower_index
    return float(lower_value + (upper_value - lower_value) * fraction)


def build_length_stats(lengths: Sequence[int]) -> dict[str, Any]:
    if not lengths:
        raise ValueError("lengths 不能为空。")

    return {
        "num_samples": len(lengths),
        "p50": percentile(lengths, 50),
        "p90": percentile(lengths, 90),
        "p95": percentile(lengths, 95),
        "p99": percentile(lengths, 99),
        "max": max(lengths),
    }


def build_output_stem(input_file: Path) -> str:
    try:
        relative_path = input_file.resolve().relative_to(REPO_ROOT)
        return "_".join(relative_path.with_suffix("").parts)
    except ValueError:
        return input_file.stem


def load_dataset(path: str | Path) -> Any:
    return load_json_file(path)


def save_dataset(path: str | Path, payload: Any) -> None:
    save_json_file(path, payload)
