"""Tests for BLEU/CHRF3 wrappers and macro aggregation."""

import json

import pytest

from gabu.evaluate import (
    evaluate_predictions,
    macro_average,
    read_predictions,
)
from gabu.metrics import corpus_scores


def test_identical_outputs_score_one_hundred():
    references = ["यह एक लाल गाड़ी है", "এটি একটি লাল গাড়ি"]
    scores = corpus_scores(references, references)
    assert scores["bleu"] == pytest.approx(100.0)
    assert scores["chrf3"] == pytest.approx(100.0)
    assert scores["n"] == 2


def test_partial_match_is_between_zero_and_hundred():
    scores = corpus_scores(["यह एक लाल गाड़ी है"], ["यह एक नीली गाड़ी है"])
    assert 0.0 <= scores["bleu"] < 100.0
    assert 0.0 <= scores["chrf3"] < 100.0


def test_length_mismatch_and_empty_inputs_raise():
    with pytest.raises(ValueError, match="same length"):
        corpus_scores(["a"], [])
    with pytest.raises(ValueError, match="empty"):
        corpus_scores([], [])


def test_macro_average_is_unweighted_language_mean():
    per_language = {
        "hi": {"bleu": 40.0, "chrf3": 60.0},
        "bn": {"bleu": 20.0, "chrf3": 50.0},
    }
    macro = macro_average(per_language)
    assert macro == {"bleu": 30.0, "chrf3": 55.0, "languages": 2}


def test_read_and_evaluate_jsonl(tmp_path):
    path = tmp_path / "hi.jsonl"
    rows = [
        {"idx": 0, "ref": "यह एक लाल गाड़ी है", "decoded": "यह एक लाल गाड़ी है"},
        {"idx": 1, "ref": "यह एक नीला घर है", "decoded": "यह एक नीला घर है"},
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    parsed = read_predictions(path)
    assert len(parsed) == 2
    scores = evaluate_predictions(path, language="hi")
    assert scores["language"] == "hi"
    assert scores["bleu"] == pytest.approx(100.0)


def test_jsonl_missing_keys_and_null_references_raise(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"idx": 0, "decoded": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing keys"):
        read_predictions(path)

    path.write_text(json.dumps({"idx": 0, "ref": None, "decoded": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no reference"):
        read_predictions(path)
