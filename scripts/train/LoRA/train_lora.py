import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train.LoRA.train_lora_config import TrainLoraConfig  # noqa: E402
from scripts.train.common.lora_training_runner import run_lora_training  # noqa: E402


def main():
    config = TrainLoraConfig.from_cli()
    run_lora_training(config, run_name="lora_train")


if __name__ == "__main__":
    main()
