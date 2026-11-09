"""Tests for the region padding, crop, resize, and coordinate pipeline."""

import pytest
from PIL import Image

from gabu.preprocess import (
    crop_region,
    normalize_box,
    pad_and_clamp_box,
    prepare_region,
    resize_thumbnail,
)


def image(width: int = 1000, height: int = 500) -> Image.Image:
    return Image.new("RGB", (width, height), (12, 34, 56))


def test_padding_expands_each_side_by_fifteen_percent():
    assert pad_and_clamp_box((100, 100, 200, 200), 1000, 500) == (85, 85, 215, 215)


def test_padding_clamps_to_image_bounds():
    assert pad_and_clamp_box((0, 0, 1000, 500), 1000, 500) == (0, 0, 1000, 500)


def test_degenerate_box_falls_back_to_full_image():
    assert pad_and_clamp_box((50, 50, 50, 80), 1000, 500) == (0, 0, 1000, 500)


def test_invalid_dimensions_and_ratio_raise():
    with pytest.raises(ValueError, match="positive"):
        pad_and_clamp_box((0, 0, 10, 10), 0, 500)
    with pytest.raises(ValueError, match="non-negative"):
        pad_and_clamp_box((0, 0, 10, 10), 100, 100, pad_ratio=-0.1)


def test_crop_returns_padded_region():
    crop, padded = crop_region(image(), (100, 100, 200, 200))
    assert padded == (85, 85, 215, 215)
    assert crop.size == (130, 130)


def test_normalize_box_uses_zero_to_one_thousand_scale():
    assert normalize_box((0, 0, 1000, 500), 1000, 500) == (0, 0, 1000, 1000)
    assert normalize_box((250, 125, 750, 375), 1000, 500) == (250, 250, 750, 750)


def test_thumbnail_preserves_aspect_ratio_and_caps_long_side():
    thumbnail = resize_thumbnail(image(1000, 500), max_side=448)
    assert thumbnail.size == (448, 224)
    assert thumbnail.mode == "RGB"


def test_prepare_region_returns_crop_and_coordinates():
    prepared = prepare_region(image(), (100, 100, 200, 200))
    assert prepared.padded_box == (85, 85, 215, 215)
    assert prepared.coordinates == (85, 170, 215, 430)
    assert prepared.image.size == (130, 130)
