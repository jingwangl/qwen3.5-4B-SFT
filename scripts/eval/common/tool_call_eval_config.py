import argparse
import os
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE_MODEL_PATH = os.environ.get(
    "QWEN_MODEL_PATH",
    str(REPO_ROOT.parent / "models" / "Qwen3.5-4B"),
)
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "test.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "base_eval"


@dataclass(frozen=True, slots=True)
class ToolCallEvalDefaults:
    description: str
    base_model_path: str = DEFAULT_BASE_MODEL_PATH
    adapter_path: Path | None = None
    data_path: Path = DEFAULT_DATA_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR


def build_parser(defaults: ToolCallEvalDefaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=defaults.description)
    parser.add_argument(
        "--model-path",
        "--base-model-path",
        dest="base_model_path",
        type=str,
        default=defaults.base_model_path,
        help="Base model path or model id.",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=defaults.adapter_path,
        help="LoRA adapter directory. Leave empty when evaluating the base model.",
    )
    parser.add_argument("--data-path", type=Path, default=defaults.data_path)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=-1,
        help="Number of samples to evaluate. Use -1 to evaluate the whole file.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=28,
        help="First-batch size for dynamic model.generate batching during evaluation.",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=None,
        help="Maximum batch size for dynamic evaluation batches. Defaults to --batch-size.",
    )
    parser.add_argument(
        "--bucket-by-length",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sort samples by prompt length before batching to reduce padding waste.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--sample-mode",
        choices=["random", "head"],
        default="random",
        help="Use random sampling or the first N rows from the dataset.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
    )
    return parser


@dataclass(slots=True)
class ToolCallEvalConfig:
    base_model_path: str
    adapter_path: Path | None
    data_path: Path
    output_dir: Path
    num_samples: int
    seed: int
    max_new_tokens: int
    batch_size: int
    max_batch_size: int
    bucket_by_length: bool
    temperature: float
    sample_mode: str
    dtype: str

    @classmethod
    def from_cli_with_defaults(cls, defaults: ToolCallEvalDefaults) -> "ToolCallEvalConfig":
        args = build_parser(defaults).parse_args()
        return cls.from_args(args)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ToolCallEvalConfig":
        if args.num_samples == 0 or args.num_samples < -1:
            raise RuntimeError("num_samples 只能是 -1 或正整数。")
        if args.max_new_tokens <= 0:
            raise RuntimeError("max_new_tokens 必须大于 0。")
        if args.batch_size <= 0:
            raise RuntimeError("batch_size 必须大于 0。")
        resolved_max_batch_size = (
            args.batch_size if args.max_batch_size is None else args.max_batch_size
        )
        if resolved_max_batch_size <= 0:
            raise RuntimeError("max_batch_size 必须大于 0。")
        if resolved_max_batch_size < args.batch_size:
            raise RuntimeError("max_batch_size 不能小于 batch_size。")
        if args.temperature < 0:
            raise RuntimeError("temperature 不能小于 0。")
        if args.adapter_path is not None and not args.adapter_path.exists():
            raise RuntimeError(f"adapter_path 不存在: {args.adapter_path}")

        return cls(
            base_model_path=args.base_model_path,
            adapter_path=args.adapter_path,
            data_path=args.data_path,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            max_batch_size=resolved_max_batch_size,
            bucket_by_length=args.bucket_by_length,
            temperature=args.temperature,
            sample_mode=args.sample_mode,
            dtype=args.dtype,
        )

    @property
    def tokenizer_path(self) -> str | Path:
        return self.adapter_path if self.adapter_path is not None else self.base_model_path

    @property
    def model_path(self) -> str:
        if self.adapter_path is not None:
            return str(self.adapter_path)
        return str(self.base_model_path)

    @property
    def model_type(self) -> str:
        return "lora_adapter" if self.adapter_path is not None else "base"

    def resolve_num_samples(self, total_rows: int) -> int:
        return total_rows if self.num_samples == -1 else min(self.num_samples, total_rows)

    def build_run_name(self, total_rows: int) -> str:
        resolved_num_samples = self.resolve_num_samples(total_rows)
        return f"n{resolved_num_samples}_seed{self.seed}_{self.sample_mode}"

    def build_summary_path(self, total_rows: int) -> Path:
        return self.output_dir / f"{self.build_run_name(total_rows)}.summary.json"

    def build_details_path(self, total_rows: int) -> Path:
        return self.output_dir / f"{self.build_run_name(total_rows)}.details.json"

    def build_summary_metadata(self, total_rows: int) -> dict:
        return {
            "run_name": self.build_run_name(total_rows),
            "model_type": self.model_type,
            "model_path": self.model_path,
            "base_model_path": str(self.base_model_path),
            "adapter_path": str(self.adapter_path) if self.adapter_path is not None else None,
            "data_path": str(self.data_path),
            "output_dir": str(self.output_dir),
            "requested_num_samples": self.num_samples,
            "resolved_num_samples": self.resolve_num_samples(total_rows),
            "seed": self.seed,
            "max_new_tokens": self.max_new_tokens,
            "batch_size": self.batch_size,
            "max_batch_size": self.max_batch_size,
            "bucket_by_length": self.bucket_by_length,
            "temperature": self.temperature,
            "sample_mode": self.sample_mode,
            "dtype": self.dtype,
        }
