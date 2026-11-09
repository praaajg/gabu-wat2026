"""Region preprocessing: 15% padding, crop, thumbnail, and 0-1000 coordinates.

These operations follow Section 3.3 and 3.4 of the paper. No object detector or
learned grounding module is used; the annotated region is the visual context.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

PAD_RATIO = 0.15
IMAGE_SIZE = 448

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class PreparedRegion:
    image: Image.Image
    padded_box: Box
    coordinates: Box


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def pad_and_clamp_box(
    box: Box,
    width: int,
    height: int,
    pad_ratio: float = PAD_RATIO,
) -> Box:
    """Expand a box by ``pad_ratio`` on each side and clamp it to the image.

    Degenerate boxes (zero or negative area) fall back to the full image, as
    described in the paper.
    """

    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    if pad_ratio < 0:
        raise ValueError("pad_ratio must be non-negative")

    x1, y1, x2, y2 = (float(value) for value in box)
    if x2 <= x1 or y2 <= y1:
        return (0, 0, width, height)

    dx = (x2 - x1) * pad_ratio
    dy = (y2 - y1) * pad_ratio
    left = int(round(_clamp(x1 - dx, 0, width)))
    top = int(round(_clamp(y1 - dy, 0, height)))
    right = int(round(_clamp(x2 + dx, 0, width)))
    bottom = int(round(_clamp(y2 + dy, 0, height)))

    if right <= left or bottom <= top:
        return (0, 0, width, height)
    return (left, top, right, bottom)


def crop_region(
    image: Image.Image,
    box: Box,
    pad_ratio: float = PAD_RATIO,
) -> tuple[Image.Image, Box]:
    """Crop the padded region and return ``(crop, padded_box)``."""

    rgb = image.convert("RGB")
    padded = pad_and_clamp_box(box, rgb.width, rgb.height, pad_ratio)
    return rgb.crop(padded), padded


def resize_thumbnail(image: Image.Image, max_side: int = IMAGE_SIZE) -> Image.Image:
    """Convert to RGB and cap the longest side while preserving aspect ratio."""

    thumbnail = image.convert("RGB").copy()
    thumbnail.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return thumbnail


def normalize_box(box: Box, width: int, height: int) -> Box:
    """Express a pixel box on the paper's 0-1000 scale."""

    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")

    x1, y1, x2, y2 = box
    left = int(round(_clamp(x1, 0, width) / width * 1000))
    top = int(round(_clamp(y1, 0, height) / height * 1000))
    right = int(round(_clamp(x2, 0, width) / width * 1000))
    bottom = int(round(_clamp(y2, 0, height) / height * 1000))
    return (left, top, right, bottom)


def prepare_region(
    image: Image.Image,
    box: Box,
    *,
    pad_ratio: float = PAD_RATIO,
    image_size: int = IMAGE_SIZE,
) -> PreparedRegion:
    """Run the complete region pipeline for one example."""

    rgb = image.convert("RGB")
    crop, padded = crop_region(rgb, box, pad_ratio)
    thumbnail = resize_thumbnail(crop, image_size)
    coordinates = normalize_box(padded, rgb.width, rgb.height)
    return PreparedRegion(
        image=thumbnail,
        padded_box=padded,
        coordinates=coordinates,
    )
