"""Tests for inference postprocessing."""

from gabu.infer import postprocess


def test_postprocess_collapses_whitespace():
    assert postprocess("  एक   लाल\nगाड़ी  ") == "एक लाल गाड़ी"


def test_postprocess_keeps_single_line():
    assert "\n" not in postprocess("line one\nline two")
