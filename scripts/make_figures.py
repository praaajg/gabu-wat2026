"""Regenerate machine-readable results and README figures for GABU."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ASSETS = ROOT / "assets"

LANGUAGE_ORDER = ["Hindi", "Bengali", "Malayalam", "Odia"]

TEST = pd.DataFrame(
    [
        {
            "language": "Hindi",
            "baseline_bleu": 31.99,
            "finetuned_bleu": 42.09,
            "delta_bleu": 10.10,
            "baseline_chrf3": 57.44,
            "finetuned_chrf3": 65.29,
            "delta_chrf3": 7.85,
        },
        {
            "language": "Bengali",
            "baseline_bleu": 22.67,
            "finetuned_bleu": 41.60,
            "delta_bleu": 18.93,
            "baseline_chrf3": 53.82,
            "finetuned_chrf3": 69.04,
            "delta_chrf3": 15.22,
        },
        {
            "language": "Malayalam",
            "baseline_bleu": 17.83,
            "finetuned_bleu": 41.68,
            "delta_bleu": 23.85,
            "baseline_chrf3": 56.55,
            "finetuned_chrf3": 72.81,
            "delta_chrf3": 16.26,
        },
        {
            "language": "Odia",
            "baseline_bleu": 5.84,
            "finetuned_bleu": 50.49,
            "delta_bleu": 44.65,
            "baseline_chrf3": 37.25,
            "finetuned_chrf3": 72.82,
            "delta_chrf3": 35.57,
        },
        {
            "language": "Macro",
            "baseline_bleu": 19.58,
            "finetuned_bleu": 43.97,
            "delta_bleu": 24.39,
            "baseline_chrf3": 51.27,
            "finetuned_chrf3": 69.99,
            "delta_chrf3": 18.72,
        },
    ]
)

CHALLENGE = pd.DataFrame(
    [
        {
            "language": "Hindi",
            "baseline_bleu": 36.51,
            "finetuned_bleu": 50.58,
            "delta_bleu": 14.07,
            "baseline_chrf3": 61.10,
            "finetuned_chrf3": 69.68,
            "delta_chrf3": 8.58,
        },
        {
            "language": "Bengali",
            "baseline_bleu": 17.61,
            "finetuned_bleu": 38.08,
            "delta_bleu": 20.47,
            "baseline_chrf3": 54.36,
            "finetuned_chrf3": 68.46,
            "delta_chrf3": 14.10,
        },
        {
            "language": "Malayalam",
            "baseline_bleu": 16.89,
            "finetuned_bleu": 32.44,
            "delta_bleu": 15.55,
            "baseline_chrf3": 53.59,
            "finetuned_chrf3": 66.62,
            "delta_chrf3": 13.03,
        },
        {
            "language": "Macro",
            "baseline_bleu": 23.67,
            "finetuned_bleu": 40.37,
            "delta_bleu": 16.70,
            "baseline_chrf3": 56.35,
            "finetuned_chrf3": 68.25,
            "delta_chrf3": 11.90,
        },
    ]
)

DEV_TRAJECTORY = pd.DataFrame(
    [
        {
            "step": 500,
            "macro_bleu": 43.70,
            "macro_chrf3": 67.20,
            "hi_bleu": 36.01,
            "hi_chrf3": 59.23,
            "bn_bleu": 36.81,
            "bn_chrf3": 66.88,
            "ml_bleu": 49.04,
            "ml_chrf3": 73.38,
            "or_bleu": 52.95,
            "or_chrf3": 69.29,
        },
        {
            "step": 1000,
            "macro_bleu": 45.82,
            "macro_chrf3": 69.31,
            "hi_bleu": 34.14,
            "hi_chrf3": 59.31,
            "bn_bleu": 43.84,
            "bn_chrf3": 70.65,
            "ml_bleu": 48.04,
            "ml_chrf3": 74.41,
            "or_bleu": 57.25,
            "or_chrf3": 72.87,
        },
        {
            "step": 1500,
            "macro_bleu": 41.70,
            "macro_chrf3": 65.43,
            "hi_bleu": 33.60,
            "hi_chrf3": 58.14,
            "bn_bleu": 45.02,
            "bn_chrf3": 68.44,
            "ml_bleu": 32.73,
            "ml_chrf3": 63.70,
            "or_bleu": 55.47,
            "or_chrf3": 71.44,
        },
        {
            "step": 2000,
            "macro_bleu": 45.74,
            "macro_chrf3": 69.09,
            "hi_bleu": 34.49,
            "hi_chrf3": 58.93,
            "bn_bleu": 43.33,
            "bn_chrf3": 72.99,
            "ml_bleu": 50.99,
            "ml_chrf3": 70.60,
            "or_bleu": 54.16,
            "or_chrf3": 73.86,
        },
        {
            "step": 2500,
            "macro_bleu": 41.67,
            "macro_chrf3": 67.03,
            "hi_bleu": 39.29,
            "hi_chrf3": 61.54,
            "bn_bleu": 42.39,
            "bn_chrf3": 74.40,
            "ml_bleu": 32.09,
            "ml_chrf3": 60.29,
            "or_bleu": 52.90,
            "or_chrf3": 71.88,
        },
        {
            "step": 3000,
            "macro_bleu": 49.76,
            "macro_chrf3": 71.59,
            "hi_bleu": 37.78,
            "hi_chrf3": 59.79,
            "bn_bleu": 49.04,
            "bn_chrf3": 75.36,
            "ml_bleu": 48.75,
            "ml_chrf3": 72.73,
            "or_bleu": 63.47,
            "or_chrf3": 78.49,
        },
    ]
)

DATASET_STATS = pd.DataFrame(
    [
        {"language": "Hindi", "train": 28930, "dev": 998, "test": 1595, "challenge": 1400},
        {"language": "Bengali", "train": 28930, "dev": 998, "test": 1595, "challenge": 1400},
        {"language": "Malayalam", "train": 28930, "dev": 998, "test": 1595, "challenge": 1400},
        {"language": "Odia", "train": 28930, "dev": 998, "test": 1595, "challenge": None},
    ]
)

PAPER_METRICS = {
    "test": {
        "macro": {"bleu": 43.97, "chrf3": 69.99},
        "baseline_macro": {"bleu": 19.58, "chrf3": 51.27},
        "delta": {"bleu": 24.39, "chrf3": 18.72},
    },
    "challenge": {
        "macro": {"bleu": 40.37, "chrf3": 68.25},
        "baseline_macro": {"bleu": 23.67, "chrf3": 56.35},
        "delta": {"bleu": 16.70, "chrf3": 11.90},
    },
    "training": {
        "train_rows": 115715,
        "dev_rows": 3992,
        "steps": 3000,
        "trainable_parameters": 139526144,
        "total_parameters": 8080626976,
        "trainable_fraction": 0.0173,
    },
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.axisbelow": True,
    }
)

BLUE = "#2f6feb"
ORANGE = "#e8772e"
GREEN = "#2da44e"
GREY = "#6e7781"


def plot_test_scores() -> None:
    frame = TEST[TEST["language"] != "Macro"]
    x = np.arange(len(frame))
    width = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

    for ax, baseline, finetuned, title in (
        (axes[0], "baseline_bleu", "finetuned_bleu", "BLEU"),
        (axes[1], "baseline_chrf3", "finetuned_chrf3", "CHRF3"),
    ):
        first = ax.bar(x - width / 2, frame[baseline], width, label="Zero-shot", color=GREY)
        second = ax.bar(x + width / 2, frame[finetuned], width, label="QLoRA", color=BLUE)
        for bars in (first, second):
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.0,
                    f"{bar.get_height():.2f}",
                    ha="center",
                    fontsize=7.5,
                )
        ax.set_xticks(x, frame["language"], rotation=15)
        ax.set_ylim(0, 85)
        ax.set_ylabel(title)
        ax.set_title(f"Test {title}")
        ax.legend(frameon=False, loc="upper left")

    fig.suptitle("GABU: zero-shot vs QLoRA on the WAT 2026 test split")
    fig.tight_layout()
    fig.savefig(ASSETS / "test_bleu_chrf3.png", dpi=180)
    plt.close(fig)


def plot_improvement() -> None:
    frame = TEST[TEST["language"] != "Macro"]
    x = np.arange(len(frame))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    first = ax.bar(x - width / 2, frame["delta_bleu"], width, label="Δ BLEU", color=GREEN)
    second = ax.bar(x + width / 2, frame["delta_chrf3"], width, label="Δ CHRF3", color=ORANGE)
    for bars in (first, second):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.6,
                f"+{bar.get_height():.2f}",
                ha="center",
                fontsize=8,
            )
    ax.set_xticks(x, frame["language"])
    ax.set_ylabel("Absolute gain")
    ax.set_title("Test gains from QLoRA adaptation")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ASSETS / "test_improvement.png", dpi=180)
    plt.close(fig)


def plot_dev_trajectory() -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.plot(
        DEV_TRAJECTORY["step"],
        DEV_TRAJECTORY["macro_bleu"],
        marker="o",
        label="Macro BLEU",
        color=BLUE,
    )
    ax.plot(
        DEV_TRAJECTORY["step"],
        DEV_TRAJECTORY["macro_chrf3"],
        marker="s",
        label="Macro CHRF3",
        color=ORANGE,
    )
    for _, row in DEV_TRAJECTORY.iterrows():
        ax.annotate(
            f"{row['macro_chrf3']:.2f}",
            (row["step"], row["macro_chrf3"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    ax.set_xlabel("Training step")
    ax.set_ylabel("Development score")
    ax.set_ylim(60, 76)
    ax.set_title("Development trajectory (macro over four languages)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ASSETS / "dev_trajectory.png", dpi=180)
    plt.close(fig)


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)

    TEST.to_csv(RESULTS / "test_results.csv", index=False)
    CHALLENGE.to_csv(RESULTS / "challenge_results.csv", index=False)
    DEV_TRAJECTORY.to_csv(RESULTS / "dev_trajectory.csv", index=False)
    DATASET_STATS.to_csv(RESULTS / "dataset_stats.csv", index=False)
    (RESULTS / "paper_metrics.json").write_text(
        json.dumps(PAPER_METRICS, indent=2) + "\n", encoding="utf-8"
    )

    plot_test_scores()
    plot_improvement()
    plot_dev_trajectory()
    print(f"Wrote results to {RESULTS} and figures to {ASSETS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
