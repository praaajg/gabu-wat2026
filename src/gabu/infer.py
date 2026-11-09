"""Resumable multilingual inference for GABU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .data import load_split_frame, resolve_image
from .model import attach_adapter, load_base_model, load_processor
from .preprocess import prepare_region
from .prompt import build_messages, get_language
from .train import read_config


def postprocess(text: str) -> str:
    """Collapse whitespace and enforce the paper's single-line output."""

    return " ".join(text.split()).strip()


def generate_translation(
    model: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    *,
    max_new_tokens: int = 32,
) -> str:
    import torch

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    prompt_length = int(inputs["input_ids"].shape[-1])

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    generated = processor.batch_decode(
        output[:, prompt_length:],
        skip_special_tokens=True,
    )[0]
    return postprocess(generated)


def _completed_indices(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    completed: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                completed.add(int(json.loads(line)["idx"]))
            except (KeyError, ValueError, json.JSONDecodeError):
                continue
    return completed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate GABU translations")
    parser.add_argument("--config", default="configs/qlora.yaml")
    parser.add_argument("--lang", required=True, help="Target language code")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test", "challenge"])
    parser.add_argument("--data-dir", help="Override data.root")
    parser.add_argument("--adapter", help="Local path or Hub id of the LoRA adapter")
    parser.add_argument("--output-dir", default="outputs/predictions")
    parser.add_argument("--limit", type=int, help="Only score the first N rows")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing output files")
    args = parser.parse_args(argv)

    config = read_config(args.config)
    language = get_language(args.lang).code
    root = Path(args.data_dir or config["data"]["root"])
    frame = load_split_frame(root, args.split, language, config)
    if args.limit:
        frame = frame.head(args.limit)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{language}.jsonl"
    text_path = output_dir / f"{language}.txt"

    completed = set() if args.no_resume else _completed_indices(jsonl_path)
    if completed:
        print(f"Resuming: {len(completed):,} rows already complete")

    processor = load_processor(config["model"].get("id"))
    model = load_base_model(
        config["model"].get("id"),
        load_in_4bit=bool(config["model"].get("load_in_4bit", True)),
        quant_type=config["model"].get("quant_type", "nf4"),
        double_quant=bool(config["model"].get("double_quant", True)),
        compute_dtype=config["model"].get("compute_dtype", "float16"),
        attn_implementation=config["model"].get("attn_implementation", "sdpa"),
    )
    if args.adapter:
        model = attach_adapter(model, args.adapter)
    model.eval()

    data = config["data"]
    max_new_tokens = int(config["inference"].get("max_new_tokens", 32))
    mode = "a" if completed and not args.no_resume else "w"

    with (
        jsonl_path.open(mode, encoding="utf-8") as jsonl_handle,
        text_path.open(mode, encoding="utf-8") as text_handle,
    ):
        for row_index, row in frame.iterrows():
            idx = int(row["idx"])
            if idx in completed:
                continue

            image = resolve_image(row[data["image_column"]], root)
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
                language,
            )
            decoded = generate_translation(
                model,
                processor,
                messages,
                max_new_tokens=max_new_tokens,
            )

            reference = None
            target_col = data["target_column"]
            if target_col in row and pd.notna(row[target_col]):
                reference = str(row[target_col])
            record = {"idx": idx, "ref": reference, "decoded": decoded}
            jsonl_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            jsonl_handle.flush()
            text_handle.write(decoded + "\n")
            text_handle.flush()

            if (row_index + 1) % 100 == 0:
                print(f"Translated {row_index + 1:,}/{len(frame):,} rows")

    print(f"Wrote {jsonl_path} and {text_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
