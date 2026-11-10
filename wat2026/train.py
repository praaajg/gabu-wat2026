from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import torch
from datasets import Dataset
from transformers import (
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from .data import LANGS, build_samples
from .prompts import build_mmt_messages, render_mmt_texts

log = logging.getLogger("wat2026.train")
DEFAULT_MODEL = "google/gemma-4-E4B-it"
LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WAT2026 EN->Indic MMT QLoRA (Gemma DDP)")
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--langs", default="hi,bn,ml,or")
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", default="outputs/wat2026")
    p.add_argument("--hub-repo-id", default=os.environ.get("HF_REPO_ID", ""))
    p.add_argument("--r", type=int, default=64)
    p.add_argument("--alpha", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="DDP eff batch = 2 workers x micro(1) x accum",
    )
    p.add_argument("--max-seq", type=int, default=1024)
    p.add_argument(
        "--thumb-px",
        type=int,
        default=448,
        help="region-crop cap in px (sprint uses 224: ~4x fewer image tokens)",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="optimizer-step budget; >0 overrides --epochs (sprint: 2500)",
    )
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument(
        "--eval-dev-n-loss",
        type=int,
        default=400,
        help="dev rows for Trainer eval_loss (fwd-only, 100/lang)",
    )
    p.add_argument(
        "--eval-dev-n-gen",
        type=int,
        default=200,
        help="dev rows for on-save generation scoring (50/lang)",
    )
    p.add_argument("--max-new", type=int, default=32)
    p.add_argument("--pad", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--time-budget-s", type=float, default=7.5 * 3600)
    p.add_argument("--resume", default="auto", help="'auto' | 'none' | <ckpt path>")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=3407)
    return p.parse_args()


def rows_for(
    data_root: str, langs: list[str], split: str, pad: float, per_lang_cap: int = 0
) -> list[dict]:
    assert split in ("train", "dev"), split
    samples, stats = build_samples(data_root, langs, split, pad)
    if not samples:
        raise RuntimeError(f"No {split} samples. Stats: {stats}")
    rows: list[dict] = []
    for lang in langs:
        lang_rows = [s for s in samples if s.lang == lang]
        if per_lang_cap > 0:
            lang_rows = lang_rows[:per_lang_cap]
        for s in lang_rows:
            rows.append(
                {
                    "lang": s.lang,
                    "image_id": s.image_id,
                    "image_path": str(s.image_path),
                    "x": s.x,
                    "y": s.y,
                    "w": s.w,
                    "h": s.h,
                    "english": s.english,
                    "target": s.target,
                }
            )
    return rows


def to_hf_dataset(rows: list[dict]) -> Dataset:
    return Dataset.from_list(rows)


class _Row:
    def __init__(self, r: dict):
        self.image_path = r["image_path"]
        self.x, self.y, self.w, self.h = (r["x"], r["y"], r["w"], r["h"])


def load_thumb(r: dict, pad: float):
    from PIL import Image

    with Image.open(r["image_path"]) as im:
        img = im.convert("RGB")
    from . import prompts as _mm
    from .data import crop_region

    crop, box = crop_region(img, r["x"], r["y"], r["w"], r["h"], pad)
    crop.thumbnail(_mm.THUMB)
    return (crop, box)


def make_collator(processor, max_seq: int, pad: float):
    tok = processor.tokenizer
    pad_id = tok.pad_token_id

    def collate(rows: list[dict]) -> dict:
        prompts, fulls, imgs = ([], [], [])
        for r in rows:
            crop, box = load_thumb(r, pad)
            pr, fu = render_mmt_texts(
                processor, r["lang"], crop, box, r["english"], r["target"]
            )
            prompts.append(pr)
            fulls.append(fu)
            imgs.append([crop])
        enc = processor(
            text=fulls,
            images=imgs,
            padding=True,
            truncation=True,
            max_length=max_seq,
            return_tensors="pt",
        )
        labels = enc["input_ids"].clone()
        labels[labels == pad_id] = -100
        for i, pr in enumerate(prompts):
            pl = len(tok(pr, truncation=True, max_length=max_seq)["input_ids"])
            labels[i, : min(pl, labels.size(1))] = -100
        enc["labels"] = labels
        return enc

    return collate


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _peft_tp_compat_shim() -> None:
    try:
        import transformers.distributed.utils as _du

        if not hasattr(_du, "initialize_tensor_parallelism"):
            _du.initialize_tensor_parallelism = lambda *a, **k: None
    except Exception:
        pass
    try:
        import transformers.integrations.tensor_parallel as _tp

        if not hasattr(_tp, "EmbeddingParallel"):
            _tp.EmbeddingParallel = type("EmbeddingParallel", (), {})
    except Exception:
        pass
    try:
        import peft.utils.save_and_load as _sl

        _sl._maybe_shard_state_dict_for_tp = lambda *a, **k: None
    except Exception:
        pass


def _prepare_kbit_lite(model):
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable()
    except Exception as e:
        log.warning("grad-ckpt enable failed (%s)", str(e)[:120])
    n_up, b_up, b_plain, b_q, n_plain = (0, 0, 0, 0, 0)
    for name, p in model.named_parameters():
        if "Params4bit" in type(p).__name__:
            b_q += p.numel() // 2
            continue
        n_plain += 1
        b_plain += p.numel() * p.element_size()
        if (
            p.numel() < 2000000
            and "norm" in name
            and (p.dtype in (torch.float16, torch.bfloat16))
        ):
            p.data = p.data.to(torch.float32)
            n_up += 1
            b_up += p.numel() * 4
    log.warning(
        "kbit-lite: 4bit=%.2fGB plain=%.2fGB (%d tensors) norms->fp32: %d (%.2fMB)",
        b_q / 1000000000.0,
        b_plain / 1000000000.0,
        n_plain,
        n_up,
        b_up / 1000000.0,
    )


def _maybe_apply_liger() -> bool:
    try:
        import liger_kernel.transformers as _lk
    except Exception as e:
        log.warning("liger not installed (%s)", str(e)[:120])
        return False
    for fn in (
        "apply_liger_kernel_to_gemma4",
        "apply_liger_kernel_to_gemma2",
        "apply_liger_kernel_to_gemma",
    ):
        try:
            getattr(_lk, fn)()
            log.warning("liger kernels applied via %s", fn)
            return True
        except Exception as e:
            log.warning("liger %s unavailable (%s)", fn, str(e)[:120])
    return False


def _patch_peft_clippable() -> None:
    try:
        import peft.tuners.lora.model as _lm

        if getattr(_lm.LoraModel._create_and_replace, "_is_patched", False):
            return
        _orig = _lm.LoraModel._create_and_replace

        def _patched(self, lora_config, adapter_name, target, *a, **k):
            if "Gemma4ClippableLinear" in str(type(target)):
                return
            try:
                return _orig(self, lora_config, adapter_name, target, *a, **k)
            except ValueError as e:
                if "Gemma4ClippableLinear" in str(e):
                    return
                raise

        _lm.LoraModel._create_and_replace = _patched
        _lm.LoraModel._create_and_replace._is_patched = True
        log.warning("peft patched: skip Gemma4ClippableLinear")
    except Exception as e:
        log.warning("peft clippable patch skipped (%s)", str(e)[:150])


def load_model_and_processor(args) -> tuple:
    from peft import LoraConfig, get_peft_model

    _maybe_apply_liger()
    try:
        from transformers import AutoModelForImageTextToText as ModelClass
    except ImportError:
        from transformers import AutoModelForMultimodalLM as ModelClass
    from transformers import AutoProcessor

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    multi = _world_size() > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    dmap = {"": local_rank} if multi else None
    try:
        processor = AutoProcessor.from_pretrained(
            args.model_id.strip(), padding_side="left", trust_remote_code=True
        )
    except TypeError:
        processor = AutoProcessor.from_pretrained(
            args.model_id.strip(), trust_remote_code=True
        )
    try:
        processor.tokenizer.padding_side = "right"
    except Exception:
        pass
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model = ModelClass.from_pretrained(
        args.model_id.strip(),
        quantization_config=bnb,
        dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map=dmap,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    _prepare_kbit_lite(model)
    _patch_peft_clippable()
    peft = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGETS,
    )
    model = get_peft_model(model, peft)
    model.is_loaded_in_4bit = True
    model.is_loaded_in_8bit = True
    if multi:
        log.warning(
            "DDP rank %d pinned to cuda:%d",
            int(os.environ.get("RANK", "0")),
            local_rank,
        )
    try:
        log.warning(
            "vram: %s",
            {
                i: f"{torch.cuda.memory_allocated(i) / 1000000000.0:.2f}GB"
                for i in range(torch.cuda.device_count())
            },
        )
    except Exception:
        pass
    return (model, processor)


def latest_local_ckpt(out: Path):
    ckpts = sorted(
        out.glob("checkpoint-*"),
        key=lambda p: (
            int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1
        ),
    )
    return str(ckpts[-1]) if ckpts else None


def resolve_checkpoint(args) -> str | None:
    if args.resume == "none":
        return None
    if args.resume != "auto":
        return args.resume
    out = Path(args.output_dir)
    ckpt = latest_local_ckpt(out)
    if ckpt:
        log.warning("resuming from local %s", ckpt)
        return ckpt
    if args.hub_repo_id:
        from huggingface_hub import snapshot_download

        try:
            local = snapshot_download(
                args.hub_repo_id, allow_patterns=["checkpoints/checkpoint-*/*"]
            )
            ckpt = latest_local_ckpt(Path(local) / "checkpoints")
            if ckpt:
                log.warning("resuming from hub %s", ckpt)
                return ckpt
        except Exception as e:
            log.warning("hub resume unavailable (%s); starting fresh", e)
    return None


class TimeLimitCallback(TrainerCallback):
    def __init__(self, budget_s: float, every: int = 25):
        self.t0 = time.monotonic()
        self.budget_s = budget_s
        self.every = every
        self.fired = False

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kw,
    ):
        if (
            state.global_step % self.every == 0
            and time.monotonic() - self.t0 > self.budget_s
        ):
            log.warning(
                "time budget hit at step %d; stopping for final save+push",
                state.global_step,
            )
            self.fired = True
            control.should_training_stop = True
        return control


class HubPushCallback(TrainerCallback):
    def __init__(self, repo_id: str, save_steps: int):
        self.repo_id = repo_id
        self.save_steps = save_steps

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kw,
    ):
        if int(os.environ.get("RANK", "0")) != 0:
            return control
        try:
            import json as _js
            import time as _t

            last = state.log_history[-1] if state.log_history else {}
            (Path(args.output_dir) / "progress_train.json").write_text(
                _js.dumps(
                    {
                        "global_step": int(state.global_step),
                        "epoch": round(float(state.epoch or 0), 4),
                        "loss": last.get("loss"),
                        "learning_rate": last.get("learning_rate"),
                        "updated": _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime()),
                    }
                )
            )
        except Exception:
            pass
        return control

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kw,
    ):
        if not self.repo_id or int(state.global_step) % self.save_steps != 0:
            return control
        if int(os.environ.get("RANK", "0")) != 0:
            return control
        self.push_now(args.output_dir, int(state.global_step))
        return control

    def push_now(self, output_dir: str, step: int) -> None:
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            api.upload_folder(
                repo_id=self.repo_id,
                folder_path=os.path.join(output_dir, f"checkpoint-{step}"),
                path_in_repo=f"checkpoints/checkpoint-{step}",
                commit_message=f"ckpt {step}",
            )
            log.warning("pushed full ckpt %d", step)
            prog = os.path.join(output_dir, "progress_train.json")
            if os.path.isfile(prog):
                api.upload_file(
                    path_or_fileobj=prog,
                    repo_id=self.repo_id,
                    path_in_repo="progress_train.json",
                    commit_message=f"progress step {step}",
                )
        except Exception as e:
            log.warning("ckpt push failed (local copy intact): %s", str(e)[:200])


class DevGenEvalCallback(TrainerCallback):
    def __init__(
        self,
        data_root: str,
        langs: list[str],
        pad: float,
        max_new: int,
        eval_n: int,
        repo_id: str,
    ):
        self.data_root = data_root
        self.langs = list(langs)
        self.pad = pad
        self.max_new = max_new
        self.eval_n = eval_n
        self.repo_id = repo_id
        self.trainer = None
        self.processor = None
        self.output_dir = ""
        self.best_chrf = -1.0
        self.best_step = -1

    def _load_best(self, out: Path) -> None:
        try:
            import json as _js

            b = _js.loads((out / "best.json").read_text())
            self.best_chrf = float(b.get("best_chrf", -1.0))
            self.best_step = int(b.get("best_step", -1))
            return
        except Exception:
            pass
        if self.repo_id:
            try:
                import json as _js

                from huggingface_hub import hf_hub_download

                p = hf_hub_download(
                    repo_id=self.repo_id, filename="best.json", repo_type="model"
                )
                b = _js.loads(Path(p).read_text())
                self.best_chrf = float(b.get("best_chrf", -1.0))
                self.best_step = int(b.get("best_step", -1))
                log.warning(
                    "best restored: step %d chrf %.2f", self.best_step, self.best_chrf
                )
            except Exception:
                pass

    def _push_file(self, local: Path, repo_path: str) -> None:
        if not self.repo_id or not local.is_file():
            return
        try:
            from huggingface_hub import HfApi

            HfApi().upload_file(
                path_or_fileobj=str(local),
                repo_id=self.repo_id,
                path_in_repo=repo_path,
                commit_message=f"dev metrics {local.name}",
            )
        except Exception as e:
            log.warning("metrics push failed (local intact): %s", str(e)[:150])

    def _push_best(self, out: Path, step: int) -> None:
        if not self.repo_id:
            return
        try:
            from huggingface_hub import HfApi

            HfApi().upload_folder(
                repo_id=self.repo_id,
                folder_path=str(out / "best_adapter"),
                path_in_repo="best_adapter",
                commit_message=f"best step {step}",
            )
            log.warning("pushed best_adapter step %d", step)
        except Exception as e:
            log.warning("best push failed (local intact): %s", str(e)[:150])

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kw,
    ):
        import json as _js

        step = int(state.global_step)
        if int(os.environ.get("RANK", "0")) != 0:
            return control
        if self.eval_n <= 0 or self.trainer is None or self.processor is None:
            return control
        out = Path(self.output_dir or args.output_dir)
        if self.best_step < 0 and self.best_chrf < 0:
            self._load_best(out)
        try:
            metrics = self._eval_dev_subset()
            metrics["step"] = step
        except Exception as e:
            log.warning("dev-subset eval failed at step %d (%s)", step, str(e)[:200])
            return control
        try:
            (out / "dev_metrics").mkdir(parents=True, exist_ok=True)
            (out / "dev_metrics" / f"metrics_ckpt-{step}.json").write_text(
                _js.dumps(metrics, indent=2, ensure_ascii=False)
            )
            hist_p = out / "dev_metrics_history.json"
            try:
                hist = _js.loads(hist_p.read_text()) if hist_p.is_file() else {}
            except Exception:
                hist = {}
            hist[str(step)] = {
                "macro_bleu": metrics["macro"]["bleu"],
                "macro_chrf": metrics["macro"]["chrf3"],
            }
            hist_p.write_text(_js.dumps(hist, indent=2))
            chrf = float(metrics["macro"]["chrf3"])
            improved = chrf > self.best_chrf
            if improved:
                self.best_chrf = chrf
                self.best_step = step
                try:
                    m = self.trainer.model
                    m = m.module if hasattr(m, "module") else m
                    m.save_pretrained(str(out / "best_adapter"))
                    self.processor.save_pretrained(str(out / "best_adapter"))
                except Exception as e:
                    log.warning("best_adapter save failed (%s)", str(e)[:150])
                else:
                    self._push_best(out, step)
            (out / "best.json").write_text(
                _js.dumps(
                    {
                        "best_step": self.best_step,
                        "best_chrf": round(self.best_chrf, 2),
                        "updated_step": step,
                    }
                )
            )
            self._push_file(
                out / "dev_metrics" / f"metrics_ckpt-{step}.json",
                f"dev_metrics/metrics_ckpt-{step}.json",
            )
            self._push_file(hist_p, "dev_metrics_history.json")
            self._push_file(out / "best.json", "best.json")
            log.warning(
                "dev-subset step %d macro BLEU=%.2f CHRF3=%.2f %s",
                step,
                metrics["macro"]["bleu"],
                metrics["macro"]["chrf3"],
                "NEW BEST"
                if improved
                else f"(best {self.best_chrf:.2f}@{self.best_step})",
            )
        except Exception as e:
            log.warning("dev metrics save failed (%s)", str(e)[:150])
        return control

    def _eval_dev_subset(self) -> dict:
        import sacrebleu

        rows = rows_for(
            self.data_root,
            self.langs,
            "dev",
            self.pad,
            per_lang_cap=max(1, self.eval_n // max(1, len(self.langs))),
        )
        m = self.trainer.model
        m = m.module if hasattr(m, "module") else m
        was_training = m.training
        m.eval()
        per_lang: dict = {}
        b_all, c_all = ([], [])
        try:
            with torch.inference_mode():
                by_lang: dict[str, list[dict]] = {}
                for r in rows:
                    by_lang.setdefault(r["lang"], []).append(r)
                for lang, lrows in by_lang.items():
                    refs, hyps = ([], [])
                    for r in lrows:
                        crop, box = load_thumb(r, self.pad)
                        msgs = build_mmt_messages(lang, crop, box, r["english"], None)
                        prompt = self.processor.apply_chat_template(
                            msgs, tokenize=False, add_generation_prompt=True
                        )
                        inp = self.processor(
                            text=[prompt], images=[[crop]], return_tensors="pt"
                        )
                        dev = next(m.parameters()).device
                        inp = {
                            k: v.to(dev, torch.float16)
                            if torch.is_floating_point(v)
                            else v.to(dev)
                            for k, v in inp.items()
                        }
                        out = m.generate(
                            **inp, max_new_tokens=self.max_new, do_sample=False
                        )
                        dec = self.processor.decode(
                            out[0][inp["input_ids"].shape[-1] :],
                            skip_special_tokens=True,
                        ).strip()
                        dec = " ".join(dec.split())
                        hyps.append(dec)
                        refs.append(r["target"])
                        del inp, out
                    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
                    chrf = sacrebleu.corpus_chrf(hyps, [refs], beta=3).score
                    per_lang[lang] = {
                        "n": len(lrows),
                        "bleu": round(bleu, 2),
                        "chrf3": round(chrf, 2),
                    }
                    b_all.append(bleu)
                    c_all.append(chrf)
                    torch.cuda.empty_cache()
        finally:
            if was_training:
                m.train()
        out = dict(per_lang)
        out["macro"] = {
            "bleu": round(sum(b_all) / max(1, len(b_all)), 2),
            "chrf3": round(sum(c_all) / max(1, len(c_all)), 2),
        }
        return out


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    _peft_tp_compat_shim()
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    args = parse_args()
    from . import prompts as _mm

    _mm.THUMB = (args.thumb_px, args.thumb_px)
    log.warning("thumb=%dpx max-seq=%d", args.thumb_px, args.max_seq)
    if args.smoke:
        args.save_steps = args.eval_steps = 25
        args.eval_dev_n_loss = min(args.eval_dev_n_loss, 40)
        args.eval_dev_n_gen = min(args.eval_dev_n_gen, 20)
    langs = [l for l in args.langs.split(",") if l in LANGS]
    assert langs, f"--langs must subset {sorted(LANGS)}"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_rows = rows_for(args.data_root, langs, "train", args.pad)
    loss_rows = rows_for(
        args.data_root,
        langs,
        "dev",
        args.pad,
        per_lang_cap=max(1, args.eval_dev_n_loss // len(langs)),
    )
    log.warning("train rows=%d dev-loss rows=%d", len(train_rows), len(loss_rows))
    train_ds = to_hf_dataset(train_rows)
    dev_ds = to_hf_dataset(loss_rows)
    model, processor = load_model_and_processor(args)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    import inspect as _insp

    _tap = set(_insp.signature(TrainingArguments.__init__).parameters)
    _eval_key = "eval_strategy" if "eval_strategy" in _tap else "evaluation_strategy"
    ta = dict(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        optim="adamw_8bit",
        logging_steps=25,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
        push_to_hub=False,
        max_steps=30 if args.smoke else args.max_steps,
    )
    ta[_eval_key] = "steps"
    ta["eval_steps"] = args.eval_steps
    ta = {k: v for k, v in ta.items() if k in _tap}
    targs = TrainingArguments(**ta)
    from transformers import EarlyStoppingCallback

    time_guard = TimeLimitCallback(args.time_budget_s)
    hub_push = HubPushCallback(args.hub_repo_id, args.save_steps)
    dev_gen = DevGenEvalCallback(
        args.data_root,
        langs,
        args.pad,
        args.max_new,
        args.eval_dev_n_gen,
        args.hub_repo_id,
    )
    dev_gen.output_dir = str(out)
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=make_collator(processor, args.max_seq, args.pad),
        callbacks=[
            time_guard,
            hub_push,
            dev_gen,
            EarlyStoppingCallback(
                early_stopping_patience=args.patience, early_stopping_threshold=0.001
            ),
        ],
    )
    dev_gen.trainer = trainer
    dev_gen.processor = processor
    trainer.train(resume_from_checkpoint=resolve_checkpoint(args))
    final_step = int(trainer.state.global_step)
    if time_guard.fired and final_step % args.save_steps != 0:
        log.warning("saving full catch-up checkpoint %d", final_step)
        trainer._save_checkpoint(trainer.model, None)
        hub_push.push_now(str(out), final_step)
    if trainer.is_world_process_zero():
        from huggingface_hub import HfApi

        best_dir = out / "best_adapter"
        if not (best_dir / "adapter_config.json").is_file():
            _restored = False
            if args.hub_repo_id:
                try:
                    import shutil as _sh

                    from huggingface_hub import snapshot_download as _snap

                    _dl = _snap(
                        repo_id=args.hub_repo_id,
                        allow_patterns=["best_adapter/*"],
                        repo_type="model",
                    )
                    _src = Path(_dl) / "best_adapter"
                    if (_src / "adapter_config.json").is_file():
                        _sh.rmtree(best_dir, ignore_errors=True)
                        _sh.copytree(_src, best_dir)
                        _restored = True
                        log.warning(
                            "restored best_adapter from Hub (kept, not overwritten)"
                        )
                except Exception as e:
                    log.warning("best restore failed (%s)", str(e)[:150])
            if not _restored:
                try:
                    _m = trainer.model
                    _m = _m.module if hasattr(_m, "module") else _m
                    _m.save_pretrained(str(best_dir))
                    processor.save_pretrained(str(best_dir))
                    log.warning(
                        "froze final weights as best_adapter (no prior best anywhere)"
                    )
                except Exception:
                    try:
                        trainer.save_model(str(best_dir))
                    except Exception as e:
                        log.warning("best fallback save failed (%s)", str(e)[:150])
        if args.hub_repo_id:
            try:
                api = HfApi()
                api.upload_folder(
                    repo_id=args.hub_repo_id,
                    folder_path=str(best_dir),
                    path_in_repo="best_adapter",
                    commit_message="best adapter",
                )
                prog = out / "progress_train.json"
                if prog.is_file():
                    api.upload_file(
                        path_or_fileobj=str(prog),
                        repo_id=args.hub_repo_id,
                        path_in_repo="progress_train.json",
                        commit_message="final progress",
                    )
                log.warning("pushed best_adapter + progress")
            except Exception as e:
                log.warning("final push failed (local intact): %s", str(e)[:200])
    log.warning("done. best adapter -> %s", out / "best_adapter")


if __name__ == "__main__":
    main()
