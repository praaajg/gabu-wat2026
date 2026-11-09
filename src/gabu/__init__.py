"""GABU: multimodal English-to-Indic translation (WAT 2026)."""

from __future__ import annotations

from .prompt import LANGUAGES, Language, get_language

__version__ = "1.0.0"
__all__ = ["LANGUAGES", "Language", "__version__", "get_language"]
