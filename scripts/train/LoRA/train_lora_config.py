import argparse
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = os.environ.get(
    "QWEN_MODEL_PATH",
    str(REPO_ROOT.parent / "models" / "Qwen3.5-4B"),
)
DEFAULT_TRAIN_FILE = REPO_ROOT / "data" / "train.json"
DEFAULT_VAL_FILE = REPO_ROOT / "data" / "val.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "lora_train_peft"
DEFAULT_TOKENIZED_CACHE_DIR = REPO_ROOT / "outputs" / "tokenized_cache"

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


@dataclass(slots=True)
class TrainLoraConfig:
    model_path: str = DEFAULT_MODEL_PATH
    train_file: Path = DEFAULT_TRAIN_FILE
    val_file: Path = DEFAULT_VAL_FILE
    output_dir: Path = DEFAULT_OUTPUT_DIR

    max_length: int = 1536
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
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
    attn_implementation: str = "flash_attention_2"

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
            target_modules=_parse_target_modules(args.target_modules),
            attn_implementation=args.attn_implementation,
            rebuild_tokenized_cache=args.rebuild_tokenized_cache,
        )
        config.validate()
        return config

    def validate(self) -> None:
        positive_ints = {
            "max_length": self.max_length,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "num_epochs": self.num_epochs,
            "log_steps": self.log_steps,
            "evals_per_epoch": self.evals_per_epoch,
            "keep_last_k_checkpoints": self.keep_last_k_checkpoints,
            "dataloader_prefetch_factor": self.dataloader_prefetch_factor,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise RuntimeError(f"{name} 必须大于 0。")

        non_negative_ints = {
            "save_steps": self.save_steps,
            "dataloader_num_workers": self.dataloader_num_workers,
        }
        for name, value in non_negative_ints.items():
            if value < 0:
                raise RuntimeError(f"{name} 不能小于 0。")

        bounded_floats = {
            "warmup_ratio": self.warmup_ratio,
            "min_lr_ratio": self.min_lr_ratio,
        }
        for name, value in bounded_floats.items():
            if not 0.0 <= value <= 1.0:
                raise RuntimeError(f"{name} 需要在 [0, 1] 之间。")

        if self.max_grad_norm <= 0:
            raise RuntimeError("max_grad_norm 必须大于 0。")
        if not self.target_modules:
            raise RuntimeError("target_modules 不能为空。")

    def build_summary(self) -> dict:
        summary = asdict(self)
        for key in ("train_file", "val_file", "output_dir", "tokenized_cache_dir"):
            summary[key] = str(summary[key])
        summary["target_modules"] = list(self.target_modules)
        return summary


def _parse_target_modules(raw_value: str) -> tuple[str, ...]:
    return tuple(module.strip() for module in raw_value.split(",") if module.strip())


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
    lora_group.add_argument(
        "--attn-implementation",
        type=str,
        choices=("sdpa", "flash_attention_2"),
        default=defaults.attn_implementation,
    )
    return parser
