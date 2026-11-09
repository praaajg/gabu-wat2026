"""Tests for target-language prompts and script constraints."""

import pytest

from gabu.prompt import (
    LANGUAGES,
    build_messages,
    build_user_text,
    get_language,
)


def test_language_codes_and_aliases():
    assert get_language("hi").name == "Hindi"
    assert get_language("Hindi").code == "hi"
    assert get_language("odia").code == "or"
    assert get_language("bangla").code == "bn"
    with pytest.raises(ValueError, match="Unsupported language"):
        get_language("fr")


def test_all_expected_languages_are_present():
    assert set(LANGUAGES) == {"hi", "bn", "ml", "or"}
    assert {language.script for language in LANGUAGES.values()} == {
        "Devanagari",
        "Bengali",
        "Malayalam",
        "Odia",
    }


def test_user_text_contains_coordinates_caption_and_script():
    text = build_user_text((10, 20, 300, 400), "A red car", "hi")
    assert "[10, 20, 300, 400]" in text
    assert "A red car" in text
    assert "Hindi" in text
    assert "Devanagari" in text


def test_messages_put_image_before_text():
    messages = build_messages("image-object", (1, 2, 3, 4), "caption", "bn")
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "Bengali" in messages[0]["content"]
    content = messages[1]["content"]
    assert content[0] == {"type": "image", "image": "image-object"}
    assert content[1]["type"] == "text"
    assert "Bengali" in content[1]["text"]


def test_every_language_builds():
    for code in LANGUAGES:
        messages = build_messages("image", (0, 0, 1000, 1000), "caption", code)
        assert messages[0]["role"] == "system"
