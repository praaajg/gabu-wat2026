"""Dataset loading for the four WAT English-to-Indic splits."""

from __future__ import annotations

import base64
import binascii
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from .preprocess import Box
from .prompt import LANGUAGES, get_language

SPLIT_KEYS = {
    "train": "train_template",
    "dev": "dev_template",
    "test": "test_template",
    "challenge": "challenge_template",
}


def resolve_image(value: Any, root: str | Path) -> Image.Image:
    """Resolve an image path, base64 payload, or PIL image to RGB."""

    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(BytesIO(bytes(value))).convert("RGB")
    if not isinstance(value, str):
        raise TypeError(f"Unsupported image value of type {type(value)!r}")

    if value.startswith("data:image"):
        _, _, value = value.partition(",")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = Path(root) / candidate
    if candidate.is_file():
        return Image.open(candidate).convert("RGB")

    try:
        payload = base64.b64decode(value, validate=False)
        return Image.open(BytesIO(payload)).convert("RGB")
    except (binascii.Error, ValueError, OSError) as exc:
        raise ValueError("Image cell is not a path, base64 payload, or valid image") from exc


def _box_from_row(row: pd.Series, box_columns: list[str]) -> Box:
    try:
        return tuple(int(round(float(row[column]))) for column in box_columns)  # type: ignore[return-value]
    except KeyError as exc:
        raise ValueError(f"Missing box column: {exc}") from exc


def load_language_frame(
    path: str | Path,
    config: dict[str, Any],
    *,
    language: str | None = None,
    require_target: bool = True,
) -> pd.DataFrame:
    """Load one language/split CSV and normalize required columns."""

    data = config["data"]
    frame = pd.read_csv(path)
    image_col = data["image_column"]
    caption_col = data["caption_column"]
    target_col = data["target_column"]
    id_col = data.get("id_column", "id")
    box_columns = list(data["box_columns"])

    required = [image_col, caption_col, *box_columns]
    if require_target:
        required.append(target_col)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    if language is not None:
        get_language(language)
        frame["language"] = get_language(language).code
    elif "language" not in frame.columns:
        raise ValueError("A language code is required when the CSV has no 'language' column")

    frame["idx"] = range(len(frame))
    frame["box"] = frame.apply(lambda row: _box_from_row(row, box_columns), axis=1)
    frame["id"] = frame[id_col] if id_col in frame.columns else frame["idx"]
    return frame.reset_index(drop=True)


def split_path(
    root: str | Path,
    split: str,
    language: str,
    config: dict[str, Any],
) -> Path:
    """Resolve the configured path for one split/language pair."""

    if split not in SPLIT_KEYS:
        raise ValueError(f"Unknown split {split!r}; expected one of {sorted(SPLIT_KEYS)}")
    template = config["data"][SPLIT_KEYS[split]]
    return Path(root) / template.format(lang=get_language(language).code)


def load_split_frame(
    root: str | Path,
    split: str,
    language: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Load one split for one language."""

    path = split_path(root, split, language, config)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {split} file for {language}: {path}")
    return load_language_frame(
        path,
        config,
        language=language,
        require_target=split != "challenge",
    )


def load_training_frame(
    root: str | Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Load and concatenate the four training languages.

    The paper trains one joint adapter on 115,715 train + 3,992 dev rows. Dev is
    included by default because those are the reported training totals; set
    ``data.include_dev_in_training`` to false to train on the train split only.
    """

    languages = [get_language(code).code for code in config["data"]["languages"]]
    frames = [load_split_frame(root, "train", language, config) for language in languages]
    if bool(config["data"].get("include_dev_in_training", True)):
        frames.extend(load_split_frame(root, "dev", language, config) for language in languages)
    return pd.concat(frames, ignore_index=True)


def load_monitor_frame(
    root: str | Path,
    config: dict[str, Any],
    *,
    per_language: int = 100,
    seed: int = 3407,
) -> pd.DataFrame:
    """Build the 100-per-language development loss subset from the paper."""

    languages = [get_language(code).code for code in config["data"]["languages"]]
    frames = []
    for language in languages:
        frame = load_split_frame(root, "dev", language, config)
        sample = frame.sample(n=min(per_language, len(frame)), random_state=seed)
        frames.append(sample)
    return pd.concat(frames, ignore_index=True)


__all__ = [
    "LANGUAGES",
    "load_language_frame",
    "load_monitor_frame",
    "load_split_frame",
    "load_training_frame",
    "resolve_image",
    "split_path",
]
