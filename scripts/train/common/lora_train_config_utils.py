from typing import Any

from scripts.common.config_utils import (
    build_dataclass_summary,
    parse_csv_items,
    validate_non_negative_int_fields,
    validate_positive_int_fields,
    validate_ratio_fields,
)


DEFAULT_TARGET_MODULES = (
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_a",
    "linear_attn.in_proj_b",
    "linear_attn.in_proj_z",
    "linear_attn.out_proj",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def parse_target_modules(raw_value: str) -> tuple[str, ...]:
    return parse_csv_items(raw_value)


def validate_lora_train_config(config: Any) -> None:
    validate_positive_int_fields(
        {
            "max_length": config.max_length,
            "per_device_train_batch_size": config.per_device_train_batch_size,
            "per_device_eval_batch_size": config.per_device_eval_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "num_epochs": config.num_epochs,
            "log_steps": config.log_steps,
            "evals_per_epoch": config.evals_per_epoch,
            "keep_last_k_checkpoints": config.keep_last_k_checkpoints,
            "dataloader_prefetch_factor": config.dataloader_prefetch_factor,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
        }
    )
    validate_non_negative_int_fields(
        {
            "save_steps": config.save_steps,
            "dataloader_num_workers": config.dataloader_num_workers,
        }
    )
    validate_ratio_fields(
        {
            "warmup_ratio": config.warmup_ratio,
            "min_lr_ratio": config.min_lr_ratio,
        }
    )

    if config.max_grad_norm <= 0:
        raise RuntimeError("max_grad_norm 必须大于 0。")
    if not config.target_modules:
        raise RuntimeError("target_modules 不能为空。")


def build_lora_train_summary(config: Any) -> dict[str, Any]:
    return build_dataclass_summary(
        config,
        path_fields=("train_file", "val_file", "output_dir", "tokenized_cache_dir"),
        tuple_fields=("target_modules",),
    )
