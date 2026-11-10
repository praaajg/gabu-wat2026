from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor

from .train import _maybe_apply_liger, _patch_peft_clippable


def get_model_class():
    try:
        from transformers import AutoModelForImageTextToText as model_class
    except ImportError:
        from transformers import AutoModelForMultimodalLM as model_class
    return model_class


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a WAT 2026 LoRA adapter into the base model"
    )
    parser.add_argument("--model-id", default="google/gemma-4-E4B-it")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--adapter-subfolder", default="best_adapter")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    marker = output / ".merge_done"
    if marker.is_file() and not args.overwrite:
        print(f"merged model already present: {output}")
        return
    _maybe_apply_liger()
    _patch_peft_clippable()
    model_class = get_model_class()
    base = model_class.from_pretrained(
        args.model_id,
        dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    adapter_kwargs = {}
    if args.adapter_subfolder:
        adapter_kwargs["subfolder"] = args.adapter_subfolder
    model = PeftModel.from_pretrained(base, args.adapter, **adapter_kwargs)
    merged = model.merge_and_unload()
    del model, base
    processor = AutoProcessor.from_pretrained(
        args.model_id, trust_remote_code=True
    )
    output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output)
    processor.save_pretrained(output)
    del merged, processor
    gc.collect()
    marker.write_text(f"merged from {args.adapter}\n", encoding="utf-8")
    print(f"merged model saved: {output}")


if __name__ == "__main__":
    main()
