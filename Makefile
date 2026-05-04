.PHONY: setup hooks lint test typecheck format check build

setup:
	uv sync

hooks:
	uv run prek install

lint:
	uv run prek run --all-files --show-diff-on-failure --color=always

test:
	uv run pytest

typecheck:
	uv run ty check src tests

format:
	uv run ruff format src tests

check:
	uv run ruff check src tests
	uv run flake8 src tests
	uv run ty check src tests
	uv run pytest

build:
	uv build
