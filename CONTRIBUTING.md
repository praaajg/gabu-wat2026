# Contributing

Thanks for your interest in this research repository. Contributions that make
the results easier to reproduce, audit, or extend are especially welcome.

## Development setup

```bash
git clone https://github.com/praaajg/gabu-wat2026.git
cd gabu-wat2026
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

## Before opening a pull request

```bash
ruff check .
pytest -q
```

- Keep changes focused and describe the motivation in the PR.
- Add or update tests for any behavioural change.
- Do not commit datasets, model weights, credentials, or generated predictions.
- If you change a reported number, update `results/` and explain the
  change in the PR. Paper results are immutable; new results belong in a new
  column or document.
- By contributing you agree that your contribution is licensed under the
  repository's MIT license.

## Reporting issues

Use the issue templates. For a reproduction problem, include the command,
Python version, GPU, package versions (`pip freeze`), and the full traceback.
