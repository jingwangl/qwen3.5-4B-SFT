import math
import time
from pathlib import Path
from typing import Any, Protocol

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from scripts.utils.common import save_json_file
from scripts.utils.train_utils import (
    DatasetStats,
    build_data_collator,
    build_peft_model,
    count_parameters,
    evaluate,
    forward_loss,
    get_dtype,
    list_trainable_lora_names,
    load_base_model,
    load_or_build_tokenized_dataset,
    prune_checkpoints,
    require_gpu,
    save_training_artifacts,
    set_seed,
    unwrap_model,
)


class LoraTrainConfigLike(Protocol):
    model_path: str
    train_file: Path
    val_file: Path
    output_dir: Path
    max_length: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    num_epochs: int
    max_train_steps: int | None
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    min_lr_ratio: float
    log_steps: int
    evals_per_epoch: int
    save_steps: int
    keep_last_k_checkpoints: int
    dataloader_num_workers: int
    dataloader_prefetch_factor: int
    max_grad_norm: float
    seed: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    target_modules: tuple[str, ...]
    attn_implementation: str
    fused_optimizer: bool
    compile_model: bool
    tokenized_cache_dir: Path
    tokenized_cache: bool
    rebuild_tokenized_cache: bool

    def build_summary(self) -> dict: ...


def print_run_info(
    config: LoraTrainConfigLike,
    device: torch.device,
    dtype: torch.dtype,
    run_name: str,
) -> None:
    print(f"run_name: {run_name}")
    print(f"device: {device}")
    print(f"device_name: {torch.cuda.get_device_name(device)}")
    print(f"dtype: {dtype}")
    print(f"model_path: {config.model_path}")
    print(f"train_file: {config.train_file}")
    print(f"val_file: {config.val_file}")
    print(f"output_dir: {config.output_dir}")
    print(f"max_length: {config.max_length}")
    print(f"num_epochs: {config.num_epochs}")
    print(f"learning_rate: {config.learning_rate}")
    print(f"min_lr_ratio: {config.min_lr_ratio}")
    print(f"max_grad_norm: {config.max_grad_norm}")
    print(f"gradient_accumulation_steps: {config.gradient_accumulation_steps}")
    print(f"evals_per_epoch: {config.evals_per_epoch}")
    print(f"save_steps: {config.save_steps}")
    print(f"dataloader_num_workers: {config.dataloader_num_workers}")
    print(f"target_modules: {list(config.target_modules)}")
    print(f"attn_implementation: {config.attn_implementation}")
    print(f"fused_optimizer: {config.fused_optimizer}")
    print(f"compile_model: {config.compile_model}")
    print(f"tokenized_cache_dir: {config.tokenized_cache_dir}")
    print(f"tokenized_cache: {config.tokenized_cache}")
    print(f"rebuild_tokenized_cache: {config.rebuild_tokenized_cache}")


def format_dataset_stats(split: str, stats: DatasetStats) -> str:
    return (
        f"{split} samples: {stats.num_samples} | "
        f"skipped: {stats.skipped_samples} | "
        f"avg_seq_len: {stats.avg_seq_len:.1f} | "
        f"max_seq_len: {stats.max_seq_len} | "
        f"target_tokens: {stats.total_target_tokens:,}"
    )


def _bytes_to_gib(value: int) -> float:
    return value / (1024 ** 3)


def get_cuda_memory_stats(device: torch.device) -> dict[str, float] | None:
    if device.type != "cuda":
        return None

    torch.cuda.synchronize(device)
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    total_bytes = properties.total_memory
    allocated_bytes = torch.cuda.memory_allocated(device)
    reserved_bytes = torch.cuda.memory_reserved(device)
    peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
    peak_reserved_bytes = torch.cuda.max_memory_reserved(device)

    return {
        "allocated_gib": _bytes_to_gib(allocated_bytes),
        "reserved_gib": _bytes_to_gib(reserved_bytes),
        "peak_allocated_gib": _bytes_to_gib(peak_allocated_bytes),
        "peak_reserved_gib": _bytes_to_gib(peak_reserved_bytes),
        "total_gib": _bytes_to_gib(total_bytes),
        "allocated_pct": (allocated_bytes / total_bytes) * 100.0,
        "reserved_pct": (reserved_bytes / total_bytes) * 100.0,
        "peak_allocated_pct": (peak_allocated_bytes / total_bytes) * 100.0,
        "peak_reserved_pct": (peak_reserved_bytes / total_bytes) * 100.0,
    }


def format_cuda_memory_stats(stats: dict[str, float] | None) -> str:
    if stats is None:
        return "gpu_mem=n/a"

    return (
        f"alloc={stats['allocated_gib']:.2f} GiB ({stats['allocated_pct']:.1f}%) | "
        f"reserved={stats['reserved_gib']:.2f} GiB ({stats['reserved_pct']:.1f}%) | "
        f"peak_alloc={stats['peak_allocated_gib']:.2f} GiB ({stats['peak_allocated_pct']:.1f}%) | "
        f"peak_reserved={stats['peak_reserved_gib']:.2f} GiB ({stats['peak_reserved_pct']:.1f}%)"
    )


def load_tokenizer(config: LoraTrainConfigLike):
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_datasets(config: LoraTrainConfigLike, tokenizer):
    train_dataset, train_stats, train_cache = load_or_build_tokenized_dataset(
        tokenizer,
        config.train_file,
        config.max_length,
        cache_dir=config.tokenized_cache_dir,
        use_cache=config.tokenized_cache,
        rebuild_cache=config.rebuild_tokenized_cache,
    )
    val_dataset, val_stats, val_cache = load_or_build_tokenized_dataset(
        tokenizer,
        config.val_file,
        config.max_length,
        cache_dir=config.tokenized_cache_dir,
        use_cache=config.tokenized_cache,
        rebuild_cache=config.rebuild_tokenized_cache,
    )

    print(format_dataset_stats("train", train_stats))
    print(format_dataset_stats("val", val_stats))

    if len(train_dataset) == 0:
        raise RuntimeError(
            "train_dataset 为空，无法训练。请检查数据格式、chat template、answers/tools 字段和 max_length。"
        )
    if len(val_dataset) == 0:
        print("警告：val_dataset 为空，后续评估会返回 None。")

    return train_dataset, val_dataset, train_stats, val_stats, {
        "train": train_cache,
        "val": val_cache,
    }


def build_dataloaders(
    config: LoraTrainConfigLike,
    tokenizer,
    train_dataset,
    val_dataset,
    device: torch.device,
):
    collate_fn = build_data_collator(
        tokenizer,
        pad_to_multiple_of=8 if device.type == "cuda" else None,
    )
    dataloader_kwargs: dict[str, Any] = {
        "num_workers": config.dataloader_num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_fn,
    }
    if config.dataloader_num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = config.dataloader_prefetch_factor

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=True,
        **dataloader_kwargs,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.per_device_eval_batch_size,
        shuffle=False,
        **dataloader_kwargs,
    )

    if len(train_dataloader) == 0:
        raise RuntimeError("train_dataloader 长度为 0，无法开始训练。")

    return train_dataloader, val_dataloader


def prepare_model(config: LoraTrainConfigLike, device: torch.device, dtype: torch.dtype):
    base_model = load_base_model(
        config.model_path,
        dtype,
        attn_implementation=config.attn_implementation,
    )
    base_model.config.use_cache = False
    actual_attn = getattr(base_model.config, "_attn_implementation", None)
    if actual_attn is None:
        actual_attn = getattr(base_model.config, "attn_implementation", None)
    if actual_attn is not None:
        print(f"resolved_attn_implementation: {actual_attn}")

    if hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()

    model, target_modules = build_peft_model(
        base_model,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
    )

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    model.to(device)
    model.train()

    trainable_params, total_params = count_parameters(model)
    if trainable_params == 0:
        raise RuntimeError("当前没有任何可训练参数，PEFT LoRA 注入失败。")

    print(f"trainable params: {trainable_params:,} / {total_params:,}")
    print(f"first trainable params: {list_trainable_lora_names(model)}")

    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    return model, target_modules, trainable_params, total_params


def create_optimizer(config: LoraTrainConfigLike, model, device: torch.device):
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer_kwargs = {
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
    }
    fused_enabled = False

    if config.fused_optimizer and device.type == "cuda":
        try:
            optimizer = AdamW(
                trainable_parameters,
                fused=True,
                **optimizer_kwargs,
            )
            fused_enabled = True
        except (RuntimeError, TypeError) as error:
            print(f"警告：fused AdamW 不可用，已回退到普通 AdamW。原因: {error}")
            optimizer = AdamW(trainable_parameters, **optimizer_kwargs)
    else:
        optimizer = AdamW(trainable_parameters, **optimizer_kwargs)

    print(f"optimizer: AdamW{' (fused)' if fused_enabled else ''}")
    return optimizer, trainable_parameters


def maybe_compile_model(config: LoraTrainConfigLike, model):
    if not config.compile_model:
        return model

    if not hasattr(torch, "compile"):
        print("警告：当前 PyTorch 不支持 torch.compile，已跳过模型编译。")
        return model

    print("torch.compile: enabled")
    return torch.compile(model)


def build_training_schedule(config: LoraTrainConfigLike, train_dataloader: DataLoader):
    updates_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    if updates_per_epoch <= 0:
        raise RuntimeError("updates_per_epoch <= 0，请检查 train_dataloader 和 gradient_accumulation_steps。")

    full_train_steps = updates_per_epoch * config.num_epochs
    if config.max_train_steps is None or config.max_train_steps <= 0:
        total_steps = full_train_steps
    else:
        total_steps = min(config.max_train_steps, full_train_steps)

    if total_steps <= 0:
        raise RuntimeError("total_steps <= 0，无法训练。")

    warmup_steps = int(total_steps * config.warmup_ratio)
    print(f"updates_per_epoch: {updates_per_epoch}")
    print(f"full_train_steps: {full_train_steps}")
    print(f"max_train_steps: {total_steps}")
    print(f"warmup_steps: {warmup_steps}")
    return updates_per_epoch, total_steps, warmup_steps


def build_eval_points(updates_in_epoch: int, evals_per_epoch: int) -> set[int]:
    if updates_in_epoch <= 0:
        return set()
    if evals_per_epoch <= 0:
        raise RuntimeError("evals_per_epoch 必须大于 0。")

    num_evals = min(evals_per_epoch, updates_in_epoch)
    return {
        math.ceil(index * updates_in_epoch / num_evals)
        for index in range(1, num_evals + 1)
    }


def update_learning_rate(
    optimizer: AdamW,
    learning_rate: float,
    min_lr_ratio: float,
    optimizer_step: int,
    total_steps: int,
    warmup_steps: int,
) -> float:
    current_step = optimizer_step + 1
    if warmup_steps > 0 and current_step <= warmup_steps:
        lr = learning_rate * float(current_step) / float(warmup_steps)
    elif total_steps <= warmup_steps:
        lr = learning_rate
    else:
        decay_steps = max(total_steps - warmup_steps - 1, 1)
        progress = float(current_step - warmup_steps - 1) / float(decay_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr_ratio = min_lr_ratio + (1.0 - min_lr_ratio) * cosine
        lr = learning_rate * lr_ratio

    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def save_best_model(config: LoraTrainConfigLike, model, tokenizer, best_eval: dict | None) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    unwrap_model(model).save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    if best_eval is not None:
        save_json_file(config.output_dir / "best_eval.json", best_eval)


def save_final_artifacts(config: LoraTrainConfigLike, summary: dict) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    save_json_file(config.output_dir / "summary.json", summary)


def maybe_save_checkpoint(
    config: LoraTrainConfigLike,
    model,
    tokenizer,
    optimizer_step: int,
    checkpoint_summary: dict,
) -> str | None:
    if config.save_steps <= 0 or optimizer_step % config.save_steps != 0:
        return None

    checkpoint_dir = save_training_artifacts(
        model=model,
        tokenizer=tokenizer,
        output_dir=config.output_dir,
        summary=checkpoint_summary,
        step=optimizer_step,
    )
    prune_checkpoints(config.output_dir, config.keep_last_k_checkpoints)
    return str(checkpoint_dir)


def run_lora_training(config: LoraTrainConfigLike, run_name: str) -> None:
    set_seed(config.seed)

    device = require_gpu()
    dtype = get_dtype()
    print_run_info(config, device, dtype, run_name)

    tokenizer = load_tokenizer(config)
    train_dataset, val_dataset, train_stats, val_stats, dataset_cache_info = load_datasets(config, tokenizer)
    train_dataloader, val_dataloader = build_dataloaders(
        config,
        tokenizer,
        train_dataset,
        val_dataset,
        device,
    )

    model, target_modules, trainable_params, total_params = prepare_model(config, device, dtype)
    optimizer, trainable_parameters = create_optimizer(config, model, device)
    model = maybe_compile_model(config, model)
    updates_per_epoch, total_steps, warmup_steps = build_training_schedule(config, train_dataloader)

    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    optimizer_step = 0
    running_loss = 0.0
    accum_steps = 0
    best_eval = None
    best_model_step = None
    saved_checkpoints = []
    log_history = []
    total_seen_tokens = 0
    total_seen_target_tokens = 0
    start_time = time.time()

    initial_memory = get_cuda_memory_stats(device)
    if initial_memory is not None:
        print(f"训练开始显存 | {format_cuda_memory_stats(initial_memory)}")

    train_step_bar = tqdm(total=total_steps, desc="训练步数")
    for epoch in range(config.num_epochs):
        remaining_steps = total_steps - optimizer_step
        if remaining_steps <= 0:
            break

        updates_in_epoch = min(updates_per_epoch, remaining_steps)
        eval_points = build_eval_points(updates_in_epoch, config.evals_per_epoch)
        epoch_optimizer_step = 0

        for step, batch in enumerate(train_dataloader, start=1):
            total_seen_tokens += int(batch["attention_mask"].sum().item())
            total_seen_target_tokens += int(batch["labels"].ne(-100).sum().item())

            loss = forward_loss(model, batch, device, dtype)
            (loss / config.gradient_accumulation_steps).backward()

            running_loss += loss.item()
            accum_steps += 1

            should_update = (
                step % config.gradient_accumulation_steps == 0
                or step == len(train_dataloader)
            )
            if not should_update:
                train_step_bar.set_postfix(
                    epoch=epoch + 1,
                    batch_step=step,
                    opt_step=optimizer_step,
                    loss=f"{loss.item():.4f}",
                    note="accumulating",
                )
                continue

            current_lr = update_learning_rate(
                optimizer,
                config.learning_rate,
                config.min_lr_ratio,
                optimizer_step,
                total_steps,
                warmup_steps,
            )
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                config.max_grad_norm,
            )
            grad_norm_value = float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            epoch_optimizer_step += 1

            avg_loss = running_loss / accum_steps
            running_loss = 0.0
            accum_steps = 0

            elapsed_seconds = max(time.time() - start_time, 1e-6)
            target_tokens_per_second = total_seen_target_tokens / elapsed_seconds

            train_step_bar.update(1)
            train_step_bar.set_postfix(
                epoch=epoch + 1,
                batch_step=step,
                opt_step=optimizer_step,
                avg_loss=f"{avg_loss:.4f}",
                lr=f"{current_lr:.2e}",
                grad=f"{grad_norm_value:.2f}",
            )

            if optimizer_step % config.log_steps == 0:
                memory_stats = get_cuda_memory_stats(device)
                log_entry = {
                    "step": optimizer_step,
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "learning_rate": current_lr,
                    "grad_norm": grad_norm_value,
                    "train_tokens_seen": total_seen_tokens,
                    "target_tokens_seen": total_seen_target_tokens,
                    "target_tokens_per_second": round(target_tokens_per_second, 2),
                }
                if memory_stats is not None:
                    log_entry.update(
                        {
                            "gpu_memory_allocated_gib": round(memory_stats["allocated_gib"], 3),
                            "gpu_memory_reserved_gib": round(memory_stats["reserved_gib"], 3),
                            "gpu_memory_peak_allocated_gib": round(memory_stats["peak_allocated_gib"], 3),
                            "gpu_memory_peak_reserved_gib": round(memory_stats["peak_reserved_gib"], 3),
                            "gpu_memory_allocated_pct": round(memory_stats["allocated_pct"], 2),
                            "gpu_memory_reserved_pct": round(memory_stats["reserved_pct"], 2),
                        }
                    )
                log_history.append(log_entry)
                train_step_bar.write(
                    f"step {optimizer_step} | epoch={epoch + 1} | "
                    f"loss={avg_loss:.4f} | lr={current_lr:.2e} | grad={grad_norm_value:.2f} | "
                    f"tok/s={target_tokens_per_second:.2f} | {format_cuda_memory_stats(memory_stats)}"
                )

            current_eval = None
            if epoch_optimizer_step in eval_points:
                print(f"\n开始验证: step={optimizer_step}")
                metrics = evaluate(model, val_dataloader, device, dtype)
                if metrics is not None:
                    metrics["epoch"] = epoch + 1
                    metrics["step"] = optimizer_step
                    log_history.append(metrics)
                    current_eval = dict(metrics)
                    print(
                        f"step {optimizer_step} | "
                        f"eval_loss={metrics['eval_loss']:.4f} | "
                        f"eval_ppl={metrics['eval_perplexity']:.2f}"
                    )
                    if best_eval is None or metrics["eval_loss"] < best_eval["eval_loss"]:
                        best_eval = dict(metrics)
                        best_model_step = optimizer_step
                        save_best_model(config, model, tokenizer, best_eval)
                        print(f"保存最佳模型: step={optimizer_step} -> {config.output_dir}")

            checkpoint_dir = maybe_save_checkpoint(
                config,
                model,
                tokenizer,
                optimizer_step,
                checkpoint_summary={
                    "step": optimizer_step,
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "learning_rate": current_lr,
                    "grad_norm": grad_norm_value,
                    "eval": current_eval,
                    "train_tokens_seen": total_seen_tokens,
                    "target_tokens_seen": total_seen_target_tokens,
                },
            )
            if checkpoint_dir is not None:
                saved_checkpoints.append(checkpoint_dir)
                print(f"保存 checkpoint: step={optimizer_step} -> {checkpoint_dir}")

            if optimizer_step >= total_steps:
                break

        if optimizer_step >= total_steps:
            break

    train_step_bar.close()

    final_metrics = evaluate(model, val_dataloader, device, dtype)
    if best_eval is None:
        best_model_step = optimizer_step
        save_best_model(config, model, tokenizer, final_metrics)

    final_memory = get_cuda_memory_stats(device)
    if final_memory is not None:
        print(f"训练结束显存 | {format_cuda_memory_stats(final_memory)}")

    elapsed_seconds = time.time() - start_time
    summary = config.build_summary()
    summary.update(
        {
            "run_name": run_name,
            "scheduler": "cosine_with_warmup",
            "trainable_params": trainable_params,
            "total_params": total_params,
            "train_dataset": train_stats.to_dict(),
            "val_dataset": val_stats.to_dict(),
            "dataset_cache": dataset_cache_info,
            "optimizer_steps": optimizer_step,
            "best_model_step": best_model_step,
            "elapsed_seconds": elapsed_seconds,
            "steps_per_second": round(optimizer_step / max(elapsed_seconds, 1e-6), 4),
            "target_tokens_per_second": round(
                total_seen_target_tokens / max(elapsed_seconds, 1e-6),
                2,
            ),
            "total_seen_tokens": total_seen_tokens,
            "total_seen_target_tokens": total_seen_target_tokens,
            "best_eval": best_eval,
            "final_eval": final_metrics,
            "log_history": log_history,
            "resolved_target_modules": list(target_modules),
            "saved_checkpoints": saved_checkpoints,
            "effective_train_batch_size": (
                config.per_device_train_batch_size * config.gradient_accumulation_steps
            ),
        }
    )
    save_final_artifacts(config, summary)

    print("训练完成")
    print(f"output_dir: {config.output_dir}")
    print(f"optimizer_steps: {optimizer_step}")
    print(f"target_tokens_seen: {total_seen_target_tokens}")
    if best_eval is not None:
        print(f"best_model_step: {best_model_step}")
        print(f"best_eval_loss: {best_eval['eval_loss']:.4f}")
        print(f"best_eval_ppl: {best_eval['eval_perplexity']:.2f}")
    if final_metrics is not None:
        print(f"final_eval_loss: {final_metrics['eval_loss']:.4f}")
        print(f"final_eval_ppl: {final_metrics['eval_perplexity']:.2f}")
