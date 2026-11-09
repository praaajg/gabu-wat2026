"""Joint four-language QLoRA training for GABU.

This reference trainer implements the protocol in Sections 4-6 of the paper.
The original training code and a public adapter URL were not released; see the
implementation note in the README. GPU dependencies are imported lazily so the
rest of the package remains testable on CPU-only machines.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .data import load_monitor_frame, load_training_frame, resolve_image
from .model import (
    DEFAULT_MODEL_ID,
    DEFAULT_TARGET_MODULES,
    build_lora_config,
    load_base_model,
    load_processor,
)
from .preprocess import prepare_region
from .prompt import build_messages


def read_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping")
    return config


class RegionDataset:
    """Torch dataset over image + box + caption -> target translation."""

    def __init__(
        self,
        frame: pd.DataFrame,
        processor: Any,
        config: dict[str, Any],
        *,
        root: str | Path,
        with_labels: bool,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.processor = processor
        self.config = config
        self.root = root
        self.with_labels = with_labels
        self.mask_prompt = bool(config["training"].get("mask_prompt_tokens", True))

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        row = self.frame.iloc[index]
        data = self.config["data"]
        image = resolve_image(row[data["image_column"]], self.root)
        region = prepare_region(
            image,
            row["box"],
            pad_ratio=float(data.get("pad_ratio", 0.15)),
            image_size=int(data.get("image_size", 448)),
        )
        messages = build_messages(
            region.image,
            region.coordinates,
            str(row[data["caption_column"]]),
            str(row["language"]),
        )

        full_messages = list(messages)
        if self.with_labels:
            reference = str(row[data["target_column"]])
            full_messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": reference}]}
            )

        encoded = self.processor.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=not self.with_labels,
            return_dict=True,
            return_tensors="pt",
        )

        item: dict[str, Any] = {}
        for key, value in encoded.items():
            if (
                isinstance(value, torch.Tensor)
                and key in {"input_ids", "attention_mask", "token_type_ids"}
                and value.ndim > 1
                and value.shape[0] == 1
            ):
                item[key] = value.squeeze(0)
            else:
                item[key] = value

        if self.with_labels:
            labels = item["input_ids"].clone()
            if self.mask_prompt:
                prompt_encoded = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                prompt_length = int(prompt_encoded["input_ids"].shape[-1])
                labels[:prompt_length] = -100
            item["labels"] = labels
        return item


def make_collator(tokenizer: Any):
    """Pad text tensors and concatenate image tensors."""

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id or 0

        max_length = max(int(item["input_ids"].shape[-1]) for item in batch)
        input_ids = torch.full((len(batch), max_length), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_length), dtype=torch.long)
        labels = torch.full((len(batch), max_length), -100, dtype=torch.long)

        for row_index, item in enumerate(batch):
            length = int(item["input_ids"].shape[-1])
            input_ids[row_index, :length] = item["input_ids"]
            attention_mask[row_index, :length] = item.get(
                "attention_mask", torch.ones(length, dtype=torch.long)
            )
            if "labels" in item:
                labels[row_index, :length] = item["labels"]

        output: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "labels" in batch[0]:
            output["labels"] = labels
        if "pixel_values" in batch[0]:
            output["pixel_values"] = torch.cat([item["pixel_values"] for item in batch], dim=0)
        if "image_sizes" in batch[0]:
            output["image_sizes"] = torch.cat([item["image_sizes"] for item in batch], dim=0)
        return output

    return collate


def build_training_arguments(config: dict[str, Any]):
    from transformers import TrainingArguments

    training = config["training"]
    kwargs: dict[str, Any] = {
        "output_dir": training["output_dir"],
        "max_steps": int(training["max_steps"]),
        "learning_rate": float(training["learning_rate"]),
        "optim": training.get("optimizer", "adamw_8bit"),
        "lr_scheduler_type": training.get("scheduler", "cosine"),
        "warmup_ratio": float(training.get("warmup_ratio", 0.05)),
        "per_device_train_batch_size": int(training["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(training.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
        "eval_strategy": "steps",
        "eval_steps": int(training["eval_steps"]),
        "save_strategy": "steps",
        "save_steps": int(training["save_steps"]),
        "logging_steps": int(training.get("logging_steps", 20)),
        "fp16": bool(training.get("fp16", True)),
        "gradient_checkpointing": bool(training.get("gradient_checkpointing", True)),
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": 3,
        "remove_unused_columns": False,
        "report_to": [],
        "seed": int(training.get("seed", 3407)),
        "data_seed": int(training.get("seed", 3407)),
        "prediction_loss_only": True,
    }
    try:
        return TrainingArguments(**kwargs)
    except TypeError:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
        return TrainingArguments(**kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the joint GABU adapter")
    parser.add_argument("--config", default="configs/qlora.yaml")
    parser.add_argument("--data-dir", help="Override data.root")
    parser.add_argument("--output-dir", help="Override training.output_dir")
    args = parser.parse_args(argv)

    config = read_config(args.config)
    root = Path(args.data_dir or config["data"]["root"])
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir

    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import EarlyStoppingCallback, Trainer, set_seed

    training = config["training"]
    lora = config["lora"]
    model_config = config["model"]
    seed = int(training.get("seed", 3407))
    set_seed(seed)

    train_frame = load_training_frame(root, config)
    monitor_frame = load_monitor_frame(root, config, per_language=100, seed=seed)
    print(f"Joint training rows: {len(train_frame):,}")
    print(f"Development monitoring rows: {len(monitor_frame):,}")

    processor = load_processor(model_config.get("id", DEFAULT_MODEL_ID))
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = load_base_model(
        model_config.get("id", DEFAULT_MODEL_ID),
        load_in_4bit=bool(model_config.get("load_in_4bit", True)),
        quant_type=model_config.get("quant_type", "nf4"),
        double_quant=bool(model_config.get("double_quant", True)),
        compute_dtype=model_config.get("compute_dtype", "float16"),
        attn_implementation=model_config.get("attn_implementation", "sdpa"),
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
    )
    model = get_peft_model(
        model,
        build_lora_config(
            r=int(lora["r"]),
            alpha=int(lora["alpha"]),
            dropout=float(lora["dropout"]),
            target_modules=lora.get("target_modules", DEFAULT_TARGET_MODULES),
            bias=lora.get("bias", "none"),
            task_type=lora.get("task_type", "CAUSAL_LM"),
        ),
    )
    model.print_trainable_parameters()

    train_dataset = RegionDataset(train_frame, processor, config, root=root, with_labels=True)
    monitor_dataset = RegionDataset(monitor_frame, processor, config, root=root, with_labels=True)
    training_args = build_training_arguments(config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=monitor_dataset,
        data_collator=make_collator(processor.tokenizer),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=int(training.get("early_stopping_patience", 3)),
                early_stopping_threshold=float(training.get("early_stopping_threshold", 0.001)),
            )
        ],
    )
    trainer.train()

    output_dir = Path(training["output_dir"])
    adapter_dir = output_dir / "best_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)

    metadata = {
        "global_step": trainer.state.global_step,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "selection_metric": "eval_loss",
        "note": (
            "The paper selected best_adapter by macro CHRF3 on a 200-row "
            "development generation subset; this reference trainer selects by "
            "development loss."
        ),
    }
    (adapter_dir / "best.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved adapter to {adapter_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
