from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Isolated single-GPU MMT eval worker")
    p.add_argument("--gpu", default="cuda:0")
    p.add_argument("--langs", default="hi,bn,ml,or")
    p.add_argument("--splits", default="test,challenge")
    p.add_argument("--data-root", required=True)
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--hub-repo", default="")
    p.add_argument("--model-id", default="google/gemma-4-E4B-it")
    p.add_argument(
        "--adapter",
        default="",
        help="optional local dir or Hub id of a LoRA adapter to score",
    )
    p.add_argument("--adapter-subfolder", default="best_adapter")
    p.add_argument(
        "--prefix",
        default="baseline",
        help="jsonl/preds filename prefix, e.g. baseline or final",
    )
    p.add_argument("--thumb-px", type=int, default=448)
    p.add_argument("--max-new", type=int, default=32)
    p.add_argument("--pad", type=float, default=0.15)
    p.add_argument("--sync-every", type=int, default=200)
    p.add_argument("--time-budget-s", type=float, default=8.5 * 3600)
    return p.parse_args()


def done_idx(jl: Path) -> set[int]:
    d: set[int] = set()
    if jl.is_file():
        with open(jl, encoding="utf-8") as f:
            for line in f:
                try:
                    d.add(int(json.loads(line)["idx"]))
                except (KeyError, ValueError, json.JSONDecodeError):
                    continue
    return d


def push(paths: list[Path], repo: str) -> None:
    if not repo:
        return
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        for p in paths:
            if p.is_file():
                api.upload_file(
                    path_or_fileobj=str(p),
                    repo_id=repo,
                    path_in_repo="eval/" + p.name,
                    commit_message=f"eval {p.name}",
                )
    except Exception as e:
        print(f"eval sync failed (local intact): {str(e)[:200]}", flush=True)


def main() -> None:
    import torch

    args = parse_args()
    t0 = time.monotonic()
    torch.cuda.set_device(args.gpu)
    from . import prompts as _mm

    _mm.THUMB = (args.thumb_px, args.thumb_px)
    from transformers import AutoProcessor, BitsAndBytesConfig

    from .data import build_samples
    from .prompts import build_mmt_messages
    from .train import _patch_peft_clippable, load_thumb

    try:
        from transformers import AutoModelForImageTextToText as _MC
    except ImportError:
        from transformers import AutoModelForMultimodalLM as _MC
    langs = [l for l in args.langs.split(",") if l]
    splits = [s for s in args.splits.split(",") if s]
    out = Path(args.eval_dir)
    out.mkdir(parents=True, exist_ok=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"[worker {args.gpu}] loading {args.model_id} ...", flush=True)
    model = _MC.from_pretrained(
        args.model_id.strip(),
        quantization_config=bnb,
        dtype=torch.float16,
        attn_implementation="sdpa",
        device_map={"": args.gpu},
        trust_remote_code=True,
    )
    if args.adapter:
        from peft import PeftModel

        _patch_peft_clippable()
        adapter_kwargs = {}
        if args.adapter_subfolder:
            adapter_kwargs["subfolder"] = args.adapter_subfolder
        model = PeftModel.from_pretrained(
            model, args.adapter.strip(), **adapter_kwargs
        )
        print(
            f"[worker {args.gpu}] adapter loaded: {args.adapter}/"
            f"{args.adapter_subfolder}",
            flush=True,
        )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        args.model_id.strip(), trust_remote_code=True
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    print(f"[worker {args.gpu}] Has generate: {hasattr(model, 'generate')}", flush=True)
    for split in splits:
        rows_all, _ = build_samples(args.data_root, langs, split, args.pad)
        for lang in langs:
            rows = [s for s in rows_all if s.lang == lang]
            if not rows:
                print(f"[worker {args.gpu}] {split}/{lang}: no text, skip", flush=True)
                continue
            jl = out / f"{args.prefix}_{split}_{lang}.jsonl"
            preds_p = out / f"{args.prefix}_{split}_{lang}_preds.txt"
            done = done_idx(jl)
            print(
                f"[worker {args.gpu}] {split}/{lang} total={len(rows)} scored={len(done)}",
                flush=True,
            )
            new = 0
            with open(jl, "a", encoding="utf-8") as f:
                for i, s in enumerate(rows):
                    if i in done:
                        continue
                    if time.monotonic() - t0 > args.time_budget_s:
                        print(
                            f"[worker {args.gpu}] TIME GUARD - rerun to continue",
                            flush=True,
                        )
                        return
                    try:
                        rd = {
                            "image_path": str(s.image_path),
                            "x": s.x,
                            "y": s.y,
                            "w": s.w,
                            "h": s.h,
                        }
                        crop, box = load_thumb(rd, args.pad)
                        msgs = build_mmt_messages(lang, crop, box, s.english, None)
                        pr = processor.apply_chat_template(
                            msgs, tokenize=False, add_generation_prompt=True
                        )
                        inp = processor(text=[pr], images=[[crop]], return_tensors="pt")
                        inp = {
                            k: v.to(args.gpu, torch.float16)
                            if torch.is_floating_point(v)
                            else v.to(args.gpu)
                            for k, v in inp.items()
                        }
                        with torch.inference_mode():
                            gen = model.generate(
                                **inp, max_new_tokens=args.max_new, do_sample=False
                            )
                        dec = processor.decode(
                            gen[0][inp["input_ids"].shape[-1] :],
                            skip_special_tokens=True,
                        ).strip()
                        dec = " ".join(dec.split())
                        del inp, gen
                    except RuntimeError as e:
                        if "out of memory" not in str(e).lower():
                            raise
                        torch.cuda.empty_cache()
                        gc.collect()
                        dec = ""
                    f.write(
                        json.dumps(
                            {
                                "idx": i,
                                "image_id": s.image_id,
                                "lang": lang,
                                "ref": s.target,
                                "decoded": dec,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    done.add(i)
                    new += 1
                    if new % 50 == 0:
                        f.flush()
                    if new % args.sync_every == 0:
                        by_idx: dict[int, str] = {}
                        with open(jl, encoding="utf-8") as rf:
                            for line in rf:
                                try:
                                    r = json.loads(line)
                                    by_idx[int(r["idx"])] = r.get("decoded", "")
                                except (KeyError, ValueError, json.JSONDecodeError):
                                    continue
                        preds_p.write_text(
                            "\n".join(by_idx.get(j, "") for j in range(len(rows))),
                            encoding="utf-8",
                        )
                        push([jl, preds_p], args.hub_repo)
            by_idx = {}
            with open(jl, encoding="utf-8") as rf:
                for line in rf:
                    try:
                        r = json.loads(line)
                        by_idx[int(r["idx"])] = r.get("decoded", "")
                    except (KeyError, ValueError, json.JSONDecodeError):
                        continue
            preds_p.write_text(
                "\n".join(by_idx.get(j, "") for j in range(len(rows))),
                encoding="utf-8",
            )
            push([jl, preds_p], args.hub_repo)
            print(
                f"[worker {args.gpu}] {split}/{lang} done scored={len(done)}/{len(rows)}",
                flush=True,
            )
    print(f"[worker {args.gpu}] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
