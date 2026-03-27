import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"
MODELS_DIR = REPO_ROOT.parent / "models"
DEFAULT_MODEL_NAME = "Qwen3.5-4B"


def get_default_model_path(model_name: str = DEFAULT_MODEL_NAME) -> str:
    return os.environ.get("QWEN_MODEL_PATH", str(MODELS_DIR / model_name))


def build_data_path(*parts: str) -> Path:
    return DATA_DIR.joinpath(*parts)


def build_output_path(*parts: str) -> Path:
    return OUTPUTS_DIR.joinpath(*parts)
