# Results artifacts

The files here are transcribed from the camera-ready paper. They are the
source of truth for README figures and are not overwritten by model runs.
`python scripts/make_figures.py` regenerates the CSVs/JSON from the same values
and rebuilds `assets/`.

| File | Contents |
|---|---|
| `test_results.csv` | Per-language test BLEU/CHRF3, baseline vs. QLoRA, deltas, plus macro. |
| `challenge_results.csv` | Challenge scores for Hindi, Bengali, Malayalam, plus macro. |
| `dev_trajectory.csv` | Development scores every 500 steps, macro and per language. |
| `dataset_stats.csv` | Train/dev/test/challenge row counts per language. |
| `paper_metrics.json` | Headline test/challenge and training-budget values. |

Notes:

- Odia challenge rows are absent because no challenge reference text was
  released.
- Development scores are generation metrics on the development set used for
  checkpoint selection; they are not held-out test scores.
- Macro values are unweighted means over the evaluated languages, rounded to
  two decimals, matching the paper.
- The paper's exact adapter artifact is not publicly linked and the evaluation
  script pulled it from the Hub head without a pinned revision. These numbers
  document the paper, not a re-run by this repository.
