"""BLEU and CHRF3 wrappers matching the paper's evaluation configuration."""

from __future__ import annotations

from collections.abc import Sequence

from sacrebleu.metrics import BLEU, CHRF


def corpus_scores(
    hypotheses: Sequence[str],
    references: Sequence[str],
    *,
    bleu_tokenize: str = "13a",
    bleu_case_sensitive: bool = True,
    chrf_beta: int = 3,
    chrf_char_order: int = 6,
    chrf_word_order: int = 2,
) -> dict[str, float | int]:
    """Compute corpus BLEU and CHRF3 for one language.

    The paper reports detokenized BLEU with 13a tokenization, case sensitivity,
    and a single reference. CHRF uses beta 3, character order 6, and word-bigram
    statistics (the CHRF++ setting).
    """

    hypotheses = list(hypotheses)
    references = list(references)
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must have the same length")
    if not hypotheses:
        raise ValueError("Cannot evaluate an empty prediction set")

    bleu = BLEU(
        tokenize=bleu_tokenize,
        lowercase=not bleu_case_sensitive,
    )
    chrf = CHRF(
        char_order=chrf_char_order,
        word_order=chrf_word_order,
        beta=chrf_beta,
    )

    bleu_score = bleu.corpus_score(hypotheses, [references]).score
    chrf_score = chrf.corpus_score(hypotheses, [references]).score
    return {
        "bleu": round(float(bleu_score), 2),
        "chrf3": round(float(chrf_score), 2),
        "n": len(hypotheses),
    }
