import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from scripts.utils.common import load_json_file, load_json_value, save_json_file


TOKENIZED_CACHE_VERSION = 1


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(slots=True)
class DatasetStats:
    num_samples: int
    skipped_samples: int
    total_tokens: int
    total_target_tokens: int
    max_seq_len: int

    @property
    def avg_seq_len(self) -> float:
        if self.num_samples == 0:
            return 0.0
        return self.total_tokens / self.num_samples

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_samples": self.num_samples,
            "skipped_samples": self.skipped_samples,
            "total_tokens": self.total_tokens,
            "total_target_tokens": self.total_target_tokens,
            "max_seq_len": self.max_seq_len,
            "avg_seq_len": round(self.avg_seq_len, 2),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DatasetStats":
        return cls(
            num_samples=int(payload["num_samples"]),
            skipped_samples=int(payload["skipped_samples"]),
            total_tokens=int(payload["total_tokens"]),
            total_target_tokens=int(payload["total_target_tokens"]),
            max_seq_len=int(payload["max_seq_len"]),
        )


def build_prompt_text(tokenizer, query: str, tools: Any) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": query}],
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_full_text(tokenizer, query: str, tools: Any, answers: Any) -> str:
    messages = [
        {"role": "user", "content": query},
        {"role": "assistant", "tool_calls": answers},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
    )


def preprocess_item(tokenizer, item: Dict[str, Any], max_length: int) -> Optional[Dict[str, Any]]:
    tools = load_json_value(item["tools"])
    answers = load_json_value(item["answers"])
    prompt_text = build_prompt_text(tokenizer, item["query"], tools)
    full_text = build_full_text(tokenizer, item["query"], tools, answers)

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]

    full_encoding = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )

    input_ids = full_encoding["input_ids"]
    attention_mask = full_encoding["attention_mask"]
    labels = input_ids[:]
    prompt_length = min(len(prompt_ids), len(labels))

    for i in range(prompt_length):
        labels[i] = -100

    target_tokens = sum(1 for label in labels if label != -100)
    if target_tokens == 0:
        return None

    return {
        "id": item.get("id"),
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "seq_len": len(input_ids),
        "target_tokens": target_tokens,
    }


class TokenizedDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        row = self.rows[idx]
        return {
            "input_ids": row["input_ids"],
            "attention_mask": row["attention_mask"],
            "labels": row["labels"],
        }


def build_dataset(
    tokenizer,
    data: List[Dict[str, Any]],
    max_length: int,
) -> Tuple[TokenizedDataset, DatasetStats]:
    rows = []
    skipped = 0
    total_tokens = 0
    total_target_tokens = 0
    max_seq_len = 0

    for item in tqdm(data, desc="预处理样本"):
        row = preprocess_item(tokenizer, item, max_length)
        if row is None:
            skipped += 1
            continue
        total_tokens += row["seq_len"]
        total_target_tokens += row["target_tokens"]
        max_seq_len = max(max_seq_len, row["seq_len"])
        row.pop("seq_len")
        row.pop("target_tokens")
        rows.append(row)

    stats = DatasetStats(
        num_samples=len(rows),
        skipped_samples=skipped,
        total_tokens=total_tokens,
        total_target_tokens=total_target_tokens,
        max_seq_len=max_seq_len,
    )
    return TokenizedDataset(rows), stats


def _normalize_cache_component(value: str) -> str:
    normalized = "".join(
        char if char.isalnum() or char in ("-", "_") else "_"
        for char in value
    ).strip("_")
    return normalized or "dataset"


def _build_tokenizer_cache_metadata(tokenizer, max_length: int) -> Dict[str, Any]:
    return {
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "padding_side": getattr(tokenizer, "padding_side", None),
        "truncation_side": getattr(tokenizer, "truncation_side", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "chat_template": getattr(tokenizer, "chat_template", None),
        "max_length": max_length,
    }


def _build_tokenized_cache_key(data_file: str | Path, tokenizer, max_length: int) -> str:
    data_path = Path(data_file).resolve()
    stat = data_path.stat()
    payload = {
        "version": TOKENIZED_CACHE_VERSION,
        "data_file": {
            "path": str(data_path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
        "tokenizer": _build_tokenizer_cache_metadata(tokenizer, max_length),
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_tokenized_cache_path(
    data_file: str | Path,
    tokenizer,
    max_length: int,
    cache_dir: str | Path,
) -> tuple[Path, str]:
    data_path = Path(data_file)
    cache_key = _build_tokenized_cache_key(data_path, tokenizer, max_length)
    cache_name = f"{_normalize_cache_component(data_path.stem)}-{cache_key[:16]}.pt"
    return Path(cache_dir) / cache_name, cache_key


def load_or_build_tokenized_dataset(
    tokenizer,
    data_file: str | Path,
    max_length: int,
    cache_dir: str | Path,
    use_cache: bool = True,
    rebuild_cache: bool = False,
) -> Tuple[TokenizedDataset, DatasetStats, Dict[str, Any]]:
    data_path = Path(data_file)
    cache_path = None
    cache_key = None
    cache_status = "disabled"

    if use_cache:
        cache_path, cache_key = build_tokenized_cache_path(
            data_path,
            tokenizer,
            max_length,
            cache_dir,
        )
        if cache_path.exists() and not rebuild_cache:
            try:
                payload = torch.load(cache_path, map_location="cpu", weights_only=False)
                rows = payload["rows"]
                stats = DatasetStats.from_dict(payload["stats"])
                print(f"tokenized cache hit: {data_path} -> {cache_path}")
                return (
                    TokenizedDataset(rows),
                    stats,
                    {
                        "enabled": True,
                        "status": "hit",
                        "cache_path": str(cache_path),
                        "cache_key": cache_key,
                    },
                )
            except Exception as error:
                print(f"警告：读取 tokenized cache 失败，准备重建。cache={cache_path} | error={error}")
                cache_status = "recovered"
        elif rebuild_cache:
            cache_status = "rebuilt"
        else:
            cache_status = "created"

    data = load_json_file(data_path)
    dataset, stats = build_dataset(tokenizer, data, max_length)

    if use_cache and cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "rows": dataset.rows,
                "stats": stats.to_dict(),
                "metadata": {
                    "version": TOKENIZED_CACHE_VERSION,
                    "data_file": str(data_path.resolve()),
                    "cache_key": cache_key,
                },
            },
            cache_path,
        )
        print(f"tokenized cache {cache_status}: {data_path} -> {cache_path}")

    return (
        dataset,
        stats,
        {
            "enabled": use_cache,
            "status": cache_status,
            "cache_path": str(cache_path) if cache_path is not None else None,
            "cache_key": cache_key,
        },
    )


def build_data_collator(
    tokenizer,
    pad_to_multiple_of: Optional[int] = None,
) -> Callable[[List[Dict[str, List[int]]]], Dict[str, torch.Tensor]]:
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise RuntimeError("tokenizer.pad_token_id 不能为空。")

    def collate_fn(features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_seq_len = max(len(feature["input_ids"]) for feature in features)
        if pad_to_multiple_of and max_seq_len % pad_to_multiple_of != 0:
            max_seq_len = math.ceil(max_seq_len / pad_to_multiple_of) * pad_to_multiple_of

        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        for feature in features:
            pad_len = max_seq_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_token_id] * pad_len)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
            batch["labels"].append(feature["labels"] + [-100] * pad_len)

        return {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in batch.items()
        }

    return collate_fn


def get_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def require_gpu() -> torch.device:
    # 训练固定使用显卡，拿不到设备就直接失败。
    if not torch.cuda.is_available():
        raise RuntimeError("训练必须使用 GPU，但当前环境没有可用的 CUDA/ROCm 设备。")
    return torch.device("cuda")


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        for key, value in batch.items()
    }


def forward_loss(model, batch: Dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    batch = move_batch_to_device(batch, device)

    if device.type == "cuda" and dtype != torch.float32:
        with torch.autocast(device_type="cuda", dtype=dtype):
            outputs = model(**batch)
    else:
        outputs = model(**batch)

    return outputs.loss


def evaluate(model, dataloader: DataLoader, device: torch.device, dtype: torch.dtype) -> Optional[Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_steps = 0

    eval_bar = tqdm(dataloader, desc="验证中", leave=False)
    with torch.no_grad():
        for batch in eval_bar:
            loss = forward_loss(model, batch, device, dtype)
            total_loss += loss.item()
            total_steps += 1
            eval_bar.set_postfix(
                eval_step=total_steps,
                avg_loss=f"{(total_loss / total_steps):.4f}",
            )

    model.train()

    if total_steps == 0:
        return None

    avg_loss = total_loss / total_steps
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return {
        "eval_loss": avg_loss,
        "eval_perplexity": perplexity,
    }


def count_parameters(model) -> Tuple[int, int]:
    trainable = 0
    total = 0

    for param in model.parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n

    return trainable, total


def build_peft_model(
    base_model,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules,
):
    target_modules = [x.strip() for x in target_modules if x.strip()]
    if not target_modules:
        raise RuntimeError("target_modules 为空。")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
        inference_mode=False,
    )

    model = get_peft_model(base_model, peft_config)
    return model, target_modules


def list_trainable_lora_names(model, max_items: int = 20) -> List[str]:
    names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            names.append(name)
        if len(names) >= max_items:
            break
    return names


def unwrap_model(model):
    current = model
    while True:
        next_model = getattr(current, "_orig_mod", None)
        if next_model is not None and next_model is not current:
            current = next_model
            continue

        next_model = getattr(current, "module", None)
        if next_model is not None and next_model is not current:
            current = next_model
            continue

        return current


def save_training_artifacts(
    model,
    tokenizer,
    output_dir: Path,
    summary: Dict[str, Any],
    step: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = output_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    unwrap_model(model).save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    save_json_file(checkpoint_dir / "summary.json", summary)
    return checkpoint_dir


def prune_checkpoints(output_dir: Path, keep_last_k: int) -> None:
    if keep_last_k <= 0:
        return

    checkpoint_dirs = []
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.rsplit("-", maxsplit=1)[1])
        except (IndexError, ValueError):
            continue
        checkpoint_dirs.append((step, path))

    checkpoint_dirs.sort(key=lambda item: item[0])
    for _, path in checkpoint_dirs[:-keep_last_k]:
        shutil.rmtree(path)


def _load_pretrained_model(
    model_path: str | Path,
    dtype: torch.dtype,
    load_kwargs: Dict[str, Any],
):
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            **load_kwargs,
        )
    except TypeError as first_error:
        try:
            return AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                **load_kwargs,
            )
        except TypeError:
            raise first_error


def load_base_model(
    model_path: str | Path,
    dtype: torch.dtype,
):
    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
    }
    return _load_pretrained_model(model_path, dtype, load_kwargs)
