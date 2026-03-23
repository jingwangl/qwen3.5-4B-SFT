from pathlib import Path

from scripts.eval.common.tool_call_eval_config import (  # noqa: E402
    ToolCallEvalConfig,
    ToolCallEvalDefaults,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "base_eval"


class EvalBaseToolCallsConfig(ToolCallEvalConfig):
    @classmethod
    def from_cli(cls) -> "EvalBaseToolCallsConfig":
        return cls.from_cli_with_defaults(
            ToolCallEvalDefaults(
                description="Evaluate the base Qwen model on tool-calling samples.",
                output_dir=DEFAULT_OUTPUT_DIR,
            )
        )
