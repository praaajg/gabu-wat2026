from __future__ import annotations

import csv
import gzip
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image

log = logging.getLogger("wat2026.data")
LANGS = {"hi": "Hindi", "bn": "Bengali", "ml": "Malayalam", "or": "Odia"}
COLS = ["image_id", "X", "Y", "W", "H", "english", "target"]
STEMS = {
    "hi": "hindi-visual-genome",
    "bn": "bengali-visual-genome",
    "ml": "malayalam-visual-genome",
    "or": "odia-visual-genome",
}
SPLIT_NAMES = {"train": "train", "dev": "dev", "test": "test", "challenge": "challenge"}
CHALLENGE_FILE = {
    "hi": "challenge-test-set",
    "bn": "challenge-test-set",
    "ml": "chtest",
}
EXT = {"hi": ".txt", "bn": ".txt", "ml": ".txt.gz", "or": ".tsv.gz"}
LANG_DIRS = {
    "hi": "hindi-visual-genome-11",
    "bn": "bengali-visual-genome-10",
    "ml": "malayalam-visual-genome-10",
    "or": "ODIA-VISUAL-GENOME",
}
IMAGE_DIR_STEMS = {
    "hi": "hindi-visual-genome",
    "bn": "bengali-visual-genome",
    "ml": "malayalam-visual-genome",
    "or": "odia-visual-genome",
}
IMAGE_DIR_SPLIT = {
    "train": "train",
    "dev": "dev",
    "test": "test",
    "challenge": "challenge-test-set",
}


@dataclass
class Sample:
    lang: str
    image_id: str
    image_path: Path
    x: int
    y: int
    w: int
    h: int
    target: str
    english: str = ""


def split_filename(lang: str, split: str) -> str:
    if split == "challenge" and lang == "or":
        raise FileNotFoundError("Odia challenge text not released (images only).")
    name = CHALLENGE_FILE[lang] if split == "challenge" else SPLIT_NAMES[split]
    return f"{STEMS[lang]}-{name}{EXT[lang]}"


def find_split_file(data_root: Path, lang: str, split: str) -> Path:
    exact = split_filename(lang, split)
    search_dirs = [data_root, data_root / lang, data_root / LANG_DIRS[lang]]
    for d in search_dirs:
        p = d / exact
        if p.is_file():
            return p
    lang_dir = data_root / LANG_DIRS[lang]
    if not lang_dir.is_dir():
        lang_dir = data_root / lang
    cands = (
        sorted(lang_dir.glob(f"*{split}*.txt"))
        + sorted(lang_dir.glob(f"*{split}*.tsv"))
        + sorted(lang_dir.glob(f"*{split}*.txt.gz"))
        + sorted(lang_dir.glob(f"*{split}*.tsv.gz"))
    )
    cands = [c for c in cands if c.is_file()]
    if len(cands) == 1:
        return cands[0]
    try:
        listing = sorted(p.name for p in data_root.iterdir())
    except OSError:
        listing = ["<unreadable>"]
    raise FileNotFoundError(
        f"No {exact} (or *{split}* fallback) for lang={lang}. Root contents: {listing}"
    )


def load_rows(path: Path) -> pd.DataFrame:
    opener = gzip.open if path.suffix == ".gz" else open
    df = pd.read_csv(
        opener(path, "rt", encoding="utf-8"),
        sep="\t",
        header=None,
        names=COLS,
        dtype=str,
        keep_default_na=False,
        engine="python",
        quoting=csv.QUOTE_NONE,
    )
    try:
        int(float(df.iloc[0]["X"]))
    except (ValueError, TypeError):
        df = df.iloc[1:].reset_index(drop=True)
    for c in ("X", "Y", "W", "H"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["X", "Y", "W", "H"])
    df = df[(df["W"] > 0) & (df["H"] > 0)]
    df["target"] = df["target"].str.strip()
    df = df[df["target"] != ""].reset_index(drop=True)
    return df


def resolve_image(image_dirs: list[Path], image_id: str) -> Path | None:
    for d in image_dirs:
        for ext in (".jpg", ".jpeg", ".png"):
            p = d / f"{image_id}{ext}"
            if p.is_file():
                return p
    return None


def crop_region(
    img: Image.Image, x: int, y: int, w: int, h: int, pad: float = 0.15
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    W, H = img.size
    x0 = max(0, int(x - w * pad))
    y0 = max(0, int(y - h * pad))
    x1 = min(W, int(x + w * (1 + pad)))
    y1 = min(H, int(y + h * (1 + pad)))
    if x1 - x0 < 8 or y1 - y0 < 8:
        x0, y0, x1, y1 = (0, 0, W, H)
    box1000 = (
        round(x0 / W * 1000),
        round(y0 / H * 1000),
        round(x1 / W * 1000),
        round(y1 / H * 1000),
    )
    return (img.crop((x0, y0, x1, y1)).convert("RGB"), box1000)


def candidate_image_dirs(data_root: Path, lang: str, split: str) -> list[Path]:
    img_split = IMAGE_DIR_SPLIT[split]
    exact = []
    for stem in dict.fromkeys(IMAGE_DIR_STEMS.values()):
        exact += [
            data_root / "hindi-visual-genome-11" / f"{stem}-{img_split}.images",
            data_root / f"{stem}-{img_split}.images",
        ]
    exact += [data_root / "images", data_root / lang / "images"]
    hit = [d for d in exact if d.is_dir()]
    if hit:
        return hit
    fb = discover_image_dirs(data_root, lang, split)
    if fb:
        return fb
    return [d for d in (data_root / lang, data_root) if d.is_dir()]


_DISCOVER_CACHE: dict = {}


def discover_image_dirs(data_root: Path, lang: str, split: str) -> list[Path]:
    key = (str(data_root), lang, split)
    if key not in _DISCOVER_CACHE:
        found: list[Path] = []
        for stem in dict.fromkeys(IMAGE_DIR_STEMS.values()):
            pat = f"{stem}-{IMAGE_DIR_SPLIT[split]}.images"
            found += sorted(p for p in data_root.rglob(pat) if p.is_dir())
        _DISCOVER_CACHE[key] = found
        if found:
            log.warning(
                "discovered image dir(s) via rglob: %s", [str(p) for p in found[:4]]
            )
        else:
            log.warning("rglob found no %s under %s", pat, data_root)
    return _DISCOVER_CACHE[key]


def build_samples(
    data_root: str | Path,
    langs: list[str],
    split: str,
    pad: float = 0.15,
    smoke_n: int = 0,
) -> tuple[list[Sample], dict]:
    data_root = Path(data_root)
    samples: list[Sample] = []
    stats: dict = {"split": split, "per_lang": {}}
    for lang in langs:
        n_missing = n_bad = 0
        try:
            df = load_rows(find_split_file(data_root, lang, split))
        except (FileNotFoundError, KeyError) as e:
            log.warning(str(e))
            stats["per_lang"][lang] = {"error": str(e)}
            continue
        img_dirs = candidate_image_dirs(data_root, lang, split)
        if smoke_n > 0:
            df = df.iloc[: max(1, smoke_n // max(1, len(langs)))]
        for _, r in df.iterrows():
            p = resolve_image(img_dirs, str(r["image_id"]))
            if p is None:
                n_missing += 1
                continue
            try:
                samples.append(
                    Sample(
                        lang,
                        str(r["image_id"]),
                        p,
                        int(r["X"]),
                        int(r["Y"]),
                        int(r["W"]),
                        int(r["H"]),
                        str(r["target"]),
                        str(r["english"]),
                    )
                )
            except (ValueError, TypeError):
                n_bad += 1
        stats["per_lang"][lang] = {
            "rows": len(df),
            "kept": sum(s.lang == lang for s in samples),
            "missing_img": n_missing,
            "bad_box": n_bad,
        }
    stats["total"] = len(samples)
    log.warning("build_samples %s: %s", split, stats)
    return (samples, stats)


def load_crop(
    sample: Sample, pad: float = 0.15
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    with Image.open(sample.image_path) as im:
        img = im.convert("RGB")
        return crop_region(img, sample.x, sample.y, sample.w, sample.h, pad)
