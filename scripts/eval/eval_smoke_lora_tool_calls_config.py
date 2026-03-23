from pathlib import Path

from scripts.eval.common.tool_call_eval_config import (  # noqa: E402
    ToolCallEvalConfig,
    ToolCallEvalDefaults,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ADAPTER_PATH = REPO_ROOT / "outputs" / "smoke_lora_train_peft"
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "smoke" / "val.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "smoke_lora_eval"


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
