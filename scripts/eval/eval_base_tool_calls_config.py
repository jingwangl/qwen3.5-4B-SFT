from scripts.eval.common.tool_call_eval_config import (  # noqa: E402
    ToolCallEvalConfig,
    ToolCallEvalDefaults,
)
from scripts.common.project_paths import build_output_path

DEFAULT_OUTPUT_DIR = build_output_path("base_eval")


class EvalBaseToolCallsConfig(ToolCallEvalConfig):
    @classmethod
    def from_cli(cls) -> "EvalBaseToolCallsConfig":
        return cls.from_cli_with_defaults(
            ToolCallEvalDefaults(
                description="Evaluate the base Qwen model on tool-calling samples.",
                output_dir=DEFAULT_OUTPUT_DIR,
            )
        )
