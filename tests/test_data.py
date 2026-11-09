"""Tests for split loading, image resolution, and column validation."""

import base64
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from gabu.data import (
    load_language_frame,
    load_split_frame,
    load_training_frame,
    resolve_image,
    split_path,
)


def png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (80, 60), (200, 10, 10)).save(buffer, format="PNG")
    return buffer.getvalue()


def config(root: str = "data", include_dev: bool = False) -> dict:
    return {
        "data": {
            "root": root,
            "train_template": "train/{lang}.csv",
            "dev_template": "dev/{lang}.csv",
            "test_template": "test/{lang}.csv",
            "challenge_template": "challenge/{lang}.csv",
            "image_column": "image",
            "caption_column": "caption",
            "target_column": "target",
            "id_column": "id",
            "box_columns": ["x1", "y1", "x2", "y2"],
            "languages": ["hi", "bn"],
            "include_dev_in_training": include_dev,
        }
    }


def write_split(root: Path, split: str, language: str, rows: int = 2) -> Path:
    path = root / split / f"{language}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "id": [f"{language}-{index}" for index in range(rows)],
            "image": ["region.png"] * rows,
            "caption": ["A red square"] * rows,
            "target": ["लाल वर्ग"] * rows,
            "x1": [10] * rows,
            "y1": [10] * rows,
            "x2": [40] * rows,
            "y2": [40] * rows,
        }
    )
    frame.to_csv(path, index=False)
    return path


def test_load_language_frame_adds_idx_box_and_language(tmp_path):
    path = write_split(tmp_path, "train", "hi")
    frame = load_language_frame(path, config(), language="hi")
    assert frame["language"].tolist() == ["hi", "hi"]
    assert frame["idx"].tolist() == [0, 1]
    assert frame["box"].tolist() == [(10, 10, 40, 40)] * 2
    assert frame["id"].tolist() == ["hi-0", "hi-1"]


def test_load_language_frame_requires_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"image": ["x"], "caption": ["y"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing required columns"):
        load_language_frame(path, config(), language="hi")


def test_split_path_and_missing_split(tmp_path):
    assert split_path(tmp_path, "test", "bn", config()).name == "bn.csv"
    with pytest.raises(ValueError, match="Unknown split"):
        split_path(tmp_path, "validation", "bn", config())
    with pytest.raises(FileNotFoundError, match="Missing test file"):
        load_split_frame(tmp_path, "test", "bn", config())


def test_load_training_frame_can_include_dev(tmp_path):
    for split in ("train", "dev"):
        for language in ("hi", "bn"):
            write_split(tmp_path, split, language)
    train_only = load_training_frame(tmp_path, config(include_dev=False))
    assert len(train_only) == 4
    with_dev = load_training_frame(tmp_path, config(include_dev=True))
    assert len(with_dev) == 8


def test_resolve_image_from_path_and_base64(tmp_path):
    image_path = tmp_path / "region.png"
    image_path.write_bytes(png_bytes())
    assert resolve_image("region.png", tmp_path).size == (80, 60)

    payload = base64.b64encode(png_bytes()).decode("ascii")
    assert resolve_image(payload, tmp_path).size == (80, 60)
    data_uri = "data:image/png;base64," + payload
    assert resolve_image(data_uri, tmp_path).size == (80, 60)


def test_resolve_image_rejects_junk(tmp_path):
    with pytest.raises(ValueError, match="not a path"):
        resolve_image("not-an-image", tmp_path)
