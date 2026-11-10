from __future__ import annotations

from .data import LANGS

THUMB = (448, 448)
SCRIPTS = {"hi": "Devanagari", "bn": "Bengali", "ml": "Malayalam", "or": "Odia"}


def mmt_system(lang: str) -> str:
    lname = LANGS[lang]
    script = SCRIPTS[lang]
    return f"You are an expert translator for image-grounded captions. You are given a cropped image region (shown) with its location, and the short English caption of that region. Translate the English caption into {lname} ({script} script), using the image to resolve any ambiguity. Output exactly one line: the {lname} caption in {script} script only — no explanation, no quotes, no English, no romanization."


def mmt_user_text(lang: str, box1000: tuple[int, int, int, int], english: str) -> str:
    lname = LANGS[lang]
    script = SCRIPTS[lang]
    return f"Region coordinates [x1, y1, x2, y2] on a 0-1000 scale: {list(box1000)}.\nEnglish caption: {english.strip()}\n{lname} caption ({script} script):"


def build_mmt_messages(
    lang: str,
    pil_image,
    box1000: tuple[int, int, int, int],
    english: str,
    target: str | None = None,
) -> list[dict]:
    assert lang in LANGS, f"unknown lang {lang!r}"
    content: list[dict] = [{"type": "text", "text": mmt_system(lang) + "\n"}]
    if pil_image is not None:
        content.append({"type": "image", "image": pil_image})
    else:
        content.append({"type": "text", "text": "\n[Region image unavailable]"})
    content.append(
        {"type": "text", "text": "\n" + mmt_user_text(lang, box1000, english)}
    )
    messages = [{"role": "user", "content": content}]
    if target is not None:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": target.strip()}]}
        )
    return messages


def render_mmt_texts(
    processor,
    lang: str,
    pil_image,
    box1000: tuple[int, int, int, int],
    english: str,
    target: str,
) -> tuple[str, str]:
    prompt = processor.apply_chat_template(
        build_mmt_messages(lang, pil_image, box1000, english, None),
        tokenize=False,
        add_generation_prompt=True,
    )
    full = processor.apply_chat_template(
        build_mmt_messages(lang, pil_image, box1000, english, target),
        tokenize=False,
        add_generation_prompt=False,
    )
    return (prompt, full)
