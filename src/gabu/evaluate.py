"""Evaluate GABU prediction JSONL files and report macro-averaged scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .metrics import corpus_scores

METRIC_KEYS = ("bleu", "chrf3")


def read_predictions(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = [key for key in ("idx", "ref", "decoded") if key not in row]
            if missing:
                raise ValueError(f"{path}:{line_number} missing keys {missing}")
            if row["ref"] is None:
                raise ValueError(f"{path}:{line_number} has no reference caption")
            rows.append(row)
    if not rows:
        raise ValueError(f"No prediction rows found in {path}")
    return rows


def evaluate_predictions(
    path: str | Path,
    *,
    language: str | None = None,
) -> dict[str, Any]:
    rows = read_predictions(path)
    scores = corpus_scores(
        [str(row["decoded"]) for row in rows],
        [str(row["ref"]) for row in rows],
    )
    result: dict[str, Any] = dict(scores)
    if language is not None:
        result["language"] = language
    return result


def macro_average(per_language: dict[str, dict[str, Any]]) -> dict[str, float | int]:
    """Unweighted mean over languages, matching the paper's aggregation."""

    if not per_language:
        raise ValueError("No language results to average")
    result: dict[str, float | int] = {}
    for key in METRIC_KEYS:
        values = [float(scores[key]) for scores in per_language.values()]
        result[key] = round(sum(values) / len(values), 2)
    result["languages"] = len(per_language)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate GABU predictions")
    parser.add_argument("--pred-dir", required=True, help="Directory with {lang}.jsonl files")
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["hi", "bn", "ml", "or"],
        help="Language codes to evaluate",
    )
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args(argv)

    prediction_dir = Path(args.pred_dir)
    per_language: dict[str, dict[str, Any]] = {}
    for language in args.languages:
        path = prediction_dir / f"{language}.jsonl"
        if not path.is_file():
            print(f"Skipping {language}: {path} not found")
            continue
        per_language[language] = evaluate_predictions(path, language=language)
        scores = per_language[language]
        print(
            f"{language}: BLEU {scores['bleu']:.2f} | CHRF3 {scores['chrf3']:.2f} | n={scores['n']}"
        )

    if not per_language:
        raise SystemExit("No prediction files found")

    result = {
        "per_language": per_language,
        "macro": macro_average(per_language),
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
