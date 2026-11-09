"""Validate the expected GABU data layout without loading images."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

SPLITS = ("train", "dev", "test", "challenge")
TEMPLATES = {
    "train": "train_template",
    "dev": "dev_template",
    "test": "test_template",
    "challenge": "challenge_template",
}


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--config", default="configs/qlora.yaml")
    parser.add_argument("--languages", nargs="+", default=["hi", "bn", "ml", "or"])
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    data = config["data"]
    required = [
        data["image_column"],
        data["caption_column"],
        *data["box_columns"],
    ]
    failures: list[str] = []
    rows: list[dict[str, object]] = []

    for split in args.splits:
        template = data[TEMPLATES[split]]
        for language in args.languages:
            path = Path(args.data_dir) / template.format(lang=language)
            if not path.is_file():
                failures.append(f"missing file: {path}")
                rows.append({"split": split, "language": language, "status": "missing"})
                continue

            frame = pd.read_csv(path, nrows=5)
            needs_target = split != "challenge"
            expected = required + ([data["target_column"]] if needs_target else [])
            missing = [column for column in expected if column not in frame.columns]
            if missing:
                failures.append(f"{path}: missing columns {missing}")
                rows.append(
                    {
                        "split": split,
                        "language": language,
                        "status": f"missing columns: {', '.join(missing)}",
                    }
                )
                continue

            rows.append(
                {
                    "split": split,
                    "language": language,
                    "status": "ok",
                    "columns": len(frame.columns),
                }
            )

    print(pd.DataFrame(rows).to_string(index=False))
    if failures:
        print("\nValidation failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nAll configured splits look valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
