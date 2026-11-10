from __future__ import annotations

import argparse
import json
from pathlib import Path

import sacrebleu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score WAT 2026 predictions")
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--prefix", default="finetuned")
    parser.add_argument("--splits", default="test,challenge")
    parser.add_argument("--langs", default="hi,bn,ml,or")
    return parser.parse_args()


def read_jsonl(path: Path) -> tuple[list[str], list[str]]:
    hypotheses = []
    references = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            hypotheses.append(row.get("decoded", ""))
            references.append(row.get("ref", ""))
    return hypotheses, references


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    langs = [lang for lang in args.langs.split(",") if lang]
    for split in [item for item in args.splits.split(",") if item]:
        metrics = {}
        for lang in langs:
            path = eval_dir / f"{args.prefix}_{split}_{lang}.jsonl"
            if not path.is_file():
                continue
            hypotheses, references = read_jsonl(path)
            if not hypotheses:
                continue
            metrics[lang] = {
                "n": len(hypotheses),
                "bleu": round(
                    sacrebleu.corpus_bleu(hypotheses, [references]).score, 2
                ),
                "chrf3": round(
                    sacrebleu.corpus_chrf(hypotheses, [references], beta=3).score, 2
                ),
            }
        scores = list(metrics.values())
        if scores:
            metrics["macro"] = {
                "bleu": round(
                    sum(item["bleu"] for item in scores) / len(scores), 2
                ),
                "chrf3": round(
                    sum(item["chrf3"] for item in scores) / len(scores), 2
                ),
            }
        output = eval_dir / f"metrics_{args.prefix}_{split}.json"
        output.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"{split}: {json.dumps(metrics, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
