from dataclasses import asdict, is_dataclass
from typing import Any, Iterable


def parse_csv_items(raw_value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw_value.split(",") if item.strip())


def validate_positive_int_fields(fields: dict[str, int]) -> None:
    for name, value in fields.items():
        if value <= 0:
            raise RuntimeError(f"{name} 必须大于 0。")


def validate_non_negative_int_fields(fields: dict[str, int]) -> None:
    for name, value in fields.items():
        if value < 0:
            raise RuntimeError(f"{name} 不能小于 0。")


def validate_ratio_fields(fields: dict[str, float]) -> None:
    for name, value in fields.items():
        if not 0.0 <= value <= 1.0:
            raise RuntimeError(f"{name} 需要在 [0, 1] 之间。")


def build_dataclass_summary(
    payload: Any,
    *,
    path_fields: Iterable[str] = (),
    tuple_fields: Iterable[str] = (),
) -> dict[str, Any]:
    if not is_dataclass(payload):
        raise TypeError("payload 必须是 dataclass 实例。")

    summary = asdict(payload)
    for field_name in path_fields:
        if field_name in summary and summary[field_name] is not None:
            summary[field_name] = str(summary[field_name])
    for field_name in tuple_fields:
        if field_name in summary and summary[field_name] is not None:
            summary[field_name] = list(summary[field_name])
    return summary
