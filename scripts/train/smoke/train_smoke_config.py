import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SMOKE_DATA_DIR = REPO_ROOT / "data" / "smoke"
DEFAULT_MODEL_PATH = os.environ.get(
    "QWEN_MODEL_PATH",
    str(REPO_ROOT.parent / "models" / "Qwen3.5-4B"),
)
DEFAULT_TRAIN_FILE = DEFAULT_SMOKE_DATA_DIR / "train.json"
DEFAULT_VAL_FILE = DEFAULT_SMOKE_DATA_DIR / "val.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "smoke_lora_train_peft"
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
        "--attn-implementation",
        type=str,
        choices=("sdpa", "flash_attention_2"),
        default="flash_attention_2",
    )
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
    attn_implementation: str
    fused_optimizer: bool
    compile_model: bool

    @classmethod
    def from_cli(cls) -> "TrainSmokeConfig":
        args = build_parser().parse_args()
        return cls.from_args(args)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainSmokeConfig":
        # 在这里把命令行字符串整理好，主脚本里直接拿结构化配置用。
        target_modules = tuple(
            module.strip()
            for module in args.target_modules.split(",")
            if module.strip()
        )
        if not target_modules:
            raise RuntimeError("target_modules 不能为空。")
        if args.max_length <= 0:
            raise RuntimeError("max_length 必须大于 0。")
        if args.per_device_train_batch_size <= 0 or args.per_device_eval_batch_size <= 0:
            raise RuntimeError("batch_size 必须大于 0。")
        if args.gradient_accumulation_steps <= 0:
            raise RuntimeError("gradient_accumulation_steps 必须大于 0。")
        if args.num_epochs <= 0:
            raise RuntimeError("num_epochs 必须大于 0。")
        if args.log_steps <= 0:
            raise RuntimeError("log_steps 必须大于 0。")
        if args.evals_per_epoch <= 0:
            raise RuntimeError("evals_per_epoch 必须大于 0。")
        if args.save_steps < 0:
            raise RuntimeError("save_steps 不能小于 0。")
        if args.keep_last_k_checkpoints <= 0:
            raise RuntimeError("keep_last_k_checkpoints 必须大于 0。")
        if args.dataloader_num_workers < 0:
            raise RuntimeError("dataloader_num_workers 不能小于 0。")
        if args.dataloader_prefetch_factor <= 0:
            raise RuntimeError("dataloader_prefetch_factor 必须大于 0。")
        if not 0.0 <= args.warmup_ratio <= 1.0:
            raise RuntimeError("warmup_ratio 需要在 [0, 1] 之间。")
        if not 0.0 <= args.min_lr_ratio <= 1.0:
            raise RuntimeError("min_lr_ratio 需要在 [0, 1] 之间。")
        if args.max_grad_norm <= 0:
            raise RuntimeError("max_grad_norm 必须大于 0。")

        return cls(
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
            target_modules=target_modules,
            attn_implementation=args.attn_implementation,
            fused_optimizer=args.fused_optimizer,
            compile_model=args.compile_model,
        )

    def build_summary(self) -> dict:
        return {
            "model_path": str(self.model_path),
            "train_file": str(self.train_file),
            "val_file": str(self.val_file),
            "output_dir": str(self.output_dir),
            "tokenized_cache_dir": str(self.tokenized_cache_dir),
            "tokenized_cache": self.tokenized_cache,
            "rebuild_tokenized_cache": self.rebuild_tokenized_cache,
            "max_length": self.max_length,
            "max_train_steps": self.max_train_steps,
            "num_epochs": self.num_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "min_lr_ratio": self.min_lr_ratio,
            "max_grad_norm": self.max_grad_norm,
            "log_steps": self.log_steps,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "evals_per_epoch": self.evals_per_epoch,
            "save_steps": self.save_steps,
            "keep_last_k_checkpoints": self.keep_last_k_checkpoints,
            "dataloader_num_workers": self.dataloader_num_workers,
            "dataloader_prefetch_factor": self.dataloader_prefetch_factor,
            "seed": self.seed,
            "target_modules": list(self.target_modules),
            "attn_implementation": self.attn_implementation,
            "fused_optimizer": self.fused_optimizer,
            "compile_model": self.compile_model,
        }
