import argparse
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train.LoRA.train_lora_config import TrainLoraConfig  # noqa: E402
from scripts.train.smoke.train_smoke_config import TrainSmokeConfig  # noqa: E402


def make_lora_args(**overrides):
    args = {
        "model_path": "dummy-model",
        "train_file": Path("/tmp/train.json"),
        "val_file": Path("/tmp/val.json"),
        "output_dir": Path("/tmp/output"),
        "max_length": 1024,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 2,
        "num_epochs": 1,
        "max_train_steps": None,
        "learning_rate": 2e-4,
        "weight_decay": 0.0,
        "warmup_ratio": 0.03,
        "min_lr_ratio": 0.1,
        "max_grad_norm": 1.0,
        "log_steps": 1,
        "evals_per_epoch": 1,
        "save_steps": 10,
        "seed": 42,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": "q_proj,v_proj",
        "rebuild_tokenized_cache": False,
    }
    args.update(overrides)
    return argparse.Namespace(**args)


def make_smoke_args(**overrides):
    args = {
        "model_path": "dummy-model",
        "train_file": Path("/tmp/train.json"),
        "val_file": Path("/tmp/val.json"),
        "output_dir": Path("/tmp/output"),
        "tokenized_cache_dir": Path("/tmp/cache"),
        "tokenized_cache": True,
        "rebuild_tokenized_cache": False,
        "max_length": 1024,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "num_epochs": 1,
        "max_train_steps": None,
        "learning_rate": 2e-4,
        "weight_decay": 0.0,
        "warmup_ratio": 0.03,
        "min_lr_ratio": 0.1,
        "max_grad_norm": 1.0,
        "log_steps": 1,
        "evals_per_epoch": 1,
        "save_steps": 10,
        "keep_last_k_checkpoints": 1,
        "dataloader_num_workers": 0,
        "dataloader_prefetch_factor": 2,
        "seed": 42,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": "q_proj,v_proj",
        "fused_optimizer": True,
        "compile_model": False,
    }
    args.update(overrides)
    return argparse.Namespace(**args)


class TrainLoraConfigTests(unittest.TestCase):
    def test_build_summary_stringifies_paths_and_lists_modules(self):
        config = TrainLoraConfig.from_args(make_lora_args())

        summary = config.build_summary()

        self.assertEqual(summary["train_file"], "/tmp/train.json")
        self.assertEqual(summary["val_file"], "/tmp/val.json")
        self.assertEqual(summary["target_modules"], ["q_proj", "v_proj"])

    def test_rejects_empty_target_modules(self):
        with self.assertRaisesRegex(RuntimeError, "target_modules 不能为空"):
            TrainLoraConfig.from_args(make_lora_args(target_modules=" , "))


class TrainSmokeConfigTests(unittest.TestCase):
    def test_parses_target_modules_from_csv(self):
        config = TrainSmokeConfig.from_args(make_smoke_args(target_modules="q_proj, v_proj"))

        self.assertEqual(config.target_modules, ("q_proj", "v_proj"))

    def test_rejects_invalid_warmup_ratio(self):
        with self.assertRaisesRegex(RuntimeError, "warmup_ratio 需要在 \\[0, 1\\] 之间"):
            TrainSmokeConfig.from_args(make_smoke_args(warmup_ratio=1.5))


if __name__ == "__main__":
    unittest.main()
