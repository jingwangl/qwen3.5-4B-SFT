import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from scripts.common.project_paths import build_data_path, build_output_path, get_default_model_path
from scripts.train.common.lora_train_config_utils import (
    DEFAULT_TARGET_MODULES,
    build_lora_train_summary,
    parse_target_modules,
    validate_lora_train_config,
)

DEFAULT_SMOKE_DATA_DIR = build_data_path("smoke")
DEFAULT_MODEL_PATH = get_default_model_path()
DEFAULT_TRAIN_FILE = DEFAULT_SMOKE_DATA_DIR / "train.json"
DEFAULT_VAL_FILE = DEFAULT_SMOKE_DATA_DIR / "val.json"
DEFAULT_OUTPUT_DIR = build_output_path("smoke_lora_train_peft")
DEFAULT_TOKENIZED_CACHE_DIR = build_output_path("tokenized_cache")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen PEFT LoRA smoke 训练脚本")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--val-file", type=Path, default=DEFAULT_VAL_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tokenized-cache-dir", type=Path, default=DEFAULT_TOKENIZED_CACHE_DIR)
    parser.add_argument(
        "--tokenized-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--rebuild-tokenized-cache",
        action="store_true",
        default=False,
    )

    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)

    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=None,
        help="最大优化步数。默认不截断，完整跑满所有 epoch 的训练数据。",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--evals-per-epoch", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--keep-last-k-checkpoints", type=int, default=1)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--dataloader-prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", type=str, default=",".join(DEFAULT_TARGET_MODULES))
    parser.add_argument(
        "--fused-optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--compile-model",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


@dataclass(slots=True)
class TrainSmokeConfig:
    model_path: str
    train_file: Path
    val_file: Path
    output_dir: Path
    tokenized_cache_dir: Path
    tokenized_cache: bool
    rebuild_tokenized_cache: bool
    max_length: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    num_epochs: int
    max_train_steps: Optional[int]
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    min_lr_ratio: float
    max_grad_norm: float
    log_steps: int
    evals_per_epoch: int
    save_steps: int
    keep_last_k_checkpoints: int
    dataloader_num_workers: int
    dataloader_prefetch_factor: int
    seed: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    target_modules: tuple[str, ...]
    fused_optimizer: bool
    compile_model: bool

    @classmethod
    def from_cli(cls) -> "TrainSmokeConfig":
        args = build_parser().parse_args()
        return cls.from_args(args)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainSmokeConfig":
        # 这里把命令行字符串整理好，主脚本里直接拿结构化配置用。
        config = cls(
            model_path=args.model_path,
            train_file=args.train_file,
            val_file=args.val_file,
            output_dir=args.output_dir,
            tokenized_cache_dir=args.tokenized_cache_dir,
            tokenized_cache=args.tokenized_cache,
            rebuild_tokenized_cache=args.rebuild_tokenized_cache,
            max_length=args.max_length,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_epochs=args.num_epochs,
            max_train_steps=args.max_train_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            min_lr_ratio=args.min_lr_ratio,
            max_grad_norm=args.max_grad_norm,
            log_steps=args.log_steps,
            evals_per_epoch=args.evals_per_epoch,
            save_steps=args.save_steps,
            keep_last_k_checkpoints=args.keep_last_k_checkpoints,
            dataloader_num_workers=args.dataloader_num_workers,
            dataloader_prefetch_factor=args.dataloader_prefetch_factor,
            seed=args.seed,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=parse_target_modules(args.target_modules),
            fused_optimizer=args.fused_optimizer,
            compile_model=args.compile_model,
        )
        validate_lora_train_config(config)
        return config

    def build_summary(self) -> dict:
        return build_lora_train_summary(self)
