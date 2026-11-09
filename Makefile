.PHONY: install dev test lint format figures clean

install:
	python -m pip install -e .

dev:
	python -m pip install -e ".[dev]"
	pre-commit install

test:
	pytest -q

lint:
	ruff check .

format:
	ruff format .

figures:
	python scripts/make_figures.py

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache

