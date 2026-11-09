"""Merge a GABU LoRA adapter into its base model for re-quantized scoring."""

from __future__ import annotations

import argparse
from pathlib import Path

from .model import DEFAULT_MODEL_ID, load_base_model, load_processor, merge_adapter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge a GABU LoRA adapter")
    parser.add_argument("--base", default=DEFAULT_MODEL_ID)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the base in 4-bit before merging (not recommended; full precision is safer)",
    )
    parser.add_argument("--compute-dtype", default="float16")
    args = parser.parse_args(argv)

    model = load_base_model(
        args.base,
        load_in_4bit=args.load_in_4bit,
        compute_dtype=args.compute_dtype,
    )
    merged = merge_adapter(model, args.adapter)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output)
    load_processor(args.base).save_pretrained(output)
    print(f"Merged adapter saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
