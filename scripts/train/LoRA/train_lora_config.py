import argparse
from dataclasses import dataclass, field
from pathlib import Path

from scripts.common.project_paths import build_data_path, build_output_path, get_default_model_path
from scripts.train.common.lora_train_config_utils import (
    DEFAULT_TARGET_MODULES,
    build_lora_train_summary,
    parse_target_modules,
    validate_lora_train_config,
)

DEFAULT_MODEL_PATH = get_default_model_path()
DEFAULT_TRAIN_FILE = build_data_path("train.json")
DEFAULT_VAL_FILE = build_data_path("val.json")
DEFAULT_OUTPUT_DIR = build_output_path("lora_train_peft")
DEFAULT_TOKENIZED_CACHE_DIR = build_output_path("tokenized_cache")


@dataclass(slots=True)
class TrainLoraConfig:
    model_path: str = DEFAULT_MODEL_PATH
    train_file: Path = DEFAULT_TRAIN_FILE
    val_file: Path = DEFAULT_VAL_FILE
    output_dir: Path = DEFAULT_OUTPUT_DIR

    max_length: int = 1536
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 3
    num_epochs: int = 3
    max_train_steps: int | None = None

    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    min_lr_ratio: float = 0.1
    max_grad_norm: float = 1.0

    log_steps: int = 2
    evals_per_epoch: int = 2
    save_steps: int = 1000
    seed: int = 42

    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = field(default_factory=lambda: DEFAULT_TARGET_MODULES)

    tokenized_cache_dir: Path = DEFAULT_TOKENIZED_CACHE_DIR
    tokenized_cache: bool = True
    rebuild_tokenized_cache: bool = False
    keep_last_k_checkpoints: int = 2
    dataloader_num_workers: int = 4
    dataloader_prefetch_factor: int = 4
    fused_optimizer: bool = True
    compile_model: bool = False

    @classmethod
    def from_cli(cls) -> "TrainLoraConfig":
        parser = build_parser(cls())
        return cls.from_args(parser.parse_args())

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainLoraConfig":
        config = cls(
            model_path=args.model_path,
            train_file=args.train_file,
            val_file=args.val_file,
            output_dir=args.output_dir,
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
            seed=args.seed,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=parse_target_modules(args.target_modules),
            rebuild_tokenized_cache=args.rebuild_tokenized_cache,
        )
        config.validate()
        return config

    def validate(self) -> None:
        validate_lora_train_config(self)

    def build_summary(self) -> dict:
        return build_lora_train_summary(self)


def build_parser(defaults: TrainLoraConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen PEFT LoRA 正式训练脚本")

    io_group = parser.add_argument_group("paths")
    io_group.add_argument("--model-path", type=str, default=defaults.model_path)
    io_group.add_argument("--train-file", type=Path, default=defaults.train_file)
    io_group.add_argument("--val-file", type=Path, default=defaults.val_file)
    io_group.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    io_group.add_argument(
        "--rebuild-tokenized-cache",
        action="store_true",
        default=defaults.rebuild_tokenized_cache,
    )

    train_group = parser.add_argument_group("training")
    train_group.add_argument("--max-length", type=int, default=defaults.max_length)
    train_group.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=defaults.per_device_train_batch_size,
    )
    train_group.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=defaults.per_device_eval_batch_size,
    )
    train_group.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=defaults.gradient_accumulation_steps,
    )
    train_group.add_argument("--num-epochs", type=int, default=defaults.num_epochs)
    train_group.add_argument(
        "--max-train-steps",
        type=int,
        default=defaults.max_train_steps,
        help="最大优化步数。默认不截断，完整跑满所有 epoch 的训练数据。",
    )
    train_group.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    train_group.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    train_group.add_argument("--warmup-ratio", type=float, default=defaults.warmup_ratio)
    train_group.add_argument("--min-lr-ratio", type=float, default=defaults.min_lr_ratio)
    train_group.add_argument("--max-grad-norm", type=float, default=defaults.max_grad_norm)
    train_group.add_argument("--log-steps", type=int, default=defaults.log_steps)
    train_group.add_argument("--evals-per-epoch", type=int, default=defaults.evals_per_epoch)
    train_group.add_argument("--save-steps", type=int, default=defaults.save_steps)
    train_group.add_argument("--seed", type=int, default=defaults.seed)

    lora_group = parser.add_argument_group("lora")
    lora_group.add_argument("--lora-rank", type=int, default=defaults.lora_rank)
    lora_group.add_argument("--lora-alpha", type=int, default=defaults.lora_alpha)
    lora_group.add_argument("--lora-dropout", type=float, default=defaults.lora_dropout)
    lora_group.add_argument(
        "--target-modules",
        type=str,
        default=",".join(defaults.target_modules),
    )
    return parser
