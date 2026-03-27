from scripts.eval.common.tool_call_eval_config import (  # noqa: E402
    ToolCallEvalConfig,
    ToolCallEvalDefaults,
)
from scripts.common.project_paths import build_data_path, build_output_path

DEFAULT_ADAPTER_PATH = build_output_path("smoke_lora_train_peft")
DEFAULT_DATA_PATH = build_data_path("smoke", "val.json")
DEFAULT_OUTPUT_DIR = build_output_path("smoke_lora_eval")


class EvalSmokeLoraToolCallsConfig(ToolCallEvalConfig):
    @classmethod
    def from_cli(cls) -> "EvalSmokeLoraToolCallsConfig":
        return cls.from_cli_with_defaults(
            ToolCallEvalDefaults(
                description="Evaluate the smoke LoRA model on tool-calling samples.",
                adapter_path=DEFAULT_ADAPTER_PATH,
                data_path=DEFAULT_DATA_PATH,
                output_dir=DEFAULT_OUTPUT_DIR,
            )
        )
