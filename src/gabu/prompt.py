"""Target-language prompt contract for GABU.

The paper specifies the prompt contract (expert-translator framing, region
coordinates, English caption, one target-script line) but does not print the
full verbatim string. The wording below implements that contract, so prompt
effects are not attributable to the paper authors' exact wording.

The paper uses one shared chat template for zero-shot inference and QLoRA
training. The system instruction names the target language and script; the user
turn carries the normalized region coordinates, the English caption, and the
translation request. The reference caption is appended as the assistant turn
during training only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    script: str


LANGUAGES: dict[str, Language] = {
    "hi": Language(code="hi", name="Hindi", script="Devanagari"),
    "bn": Language(code="bn", name="Bengali", script="Bengali"),
    "ml": Language(code="ml", name="Malayalam", script="Malayalam"),
    "or": Language(code="or", name="Odia", script="Odia"),
}

_ALIASES = {
    "hindi": "hi",
    "bengali": "bn",
    "bangla": "bn",
    "malayalam": "ml",
    "odia": "or",
    "oriya": "or",
}

SYSTEM_INSTRUCTION = (
    "You are an expert translator. You are given a cropped image region, its "
    "location as [x1, y1, x2, y2] on a 0-1000 scale, and a short English caption "
    "describing that region. Translate the caption into {language}. Output exactly "
    "one line containing only the {language} translation in {script} script, with "
    "no explanation, quotes, English text, or romanization."
)


def get_language(code: str) -> Language:
    """Resolve a language code or English name to a supported language."""

    normalized = code.strip().lower()
    normalized = _ALIASES.get(normalized, normalized)
    try:
        return LANGUAGES[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported language {code!r}; expected one of {sorted(LANGUAGES)}"
        ) from exc


def build_user_text(box: Box, caption: str, language: str) -> str:
    """Render the user turn after the region image."""

    target = get_language(language)
    x1, y1, x2, y2 = box
    return (
        f"Region coordinates: [{x1}, {y1}, {x2}, {y2}] on a 0-1000 scale.\n"
        f"English caption: {caption}\n"
        f"Translate the caption into {target.name}. Output only the {target.name} "
        f"translation in {target.script} script."
    )


def build_messages(
    image: Any,
    box: Box,
    caption: str,
    language: str,
) -> list[dict[str, Any]]:
    """Build the system + image + user message sequence used everywhere."""

    target = get_language(language)
    system_text = SYSTEM_INSTRUCTION.format(
        language=target.name,
        script=target.script,
    )
    return [
        {"role": "system", "content": system_text},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_user_text(box, caption, target.code)},
            ],
        },
    ]
