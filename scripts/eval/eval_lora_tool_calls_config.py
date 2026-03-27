from scripts.eval.common.tool_call_eval_config import (  # noqa: E402
    ToolCallEvalConfig,
    ToolCallEvalDefaults,
)
from scripts.common.project_paths import build_output_path

DEFAULT_ADAPTER_PATH = build_output_path("lora_train_peft")
DEFAULT_OUTPUT_DIR = build_output_path("lora_eval")


class EvalLoraToolCallsConfig(ToolCallEvalConfig):
    @classmethod
    def from_cli(cls) -> "EvalLoraToolCallsConfig":
        return cls.from_cli_with_defaults(
            ToolCallEvalDefaults(
                description="Evaluate the formal LoRA model on tool-calling samples.",
                adapter_path=DEFAULT_ADAPTER_PATH,
                output_dir=DEFAULT_OUTPUT_DIR,
            )
        )
