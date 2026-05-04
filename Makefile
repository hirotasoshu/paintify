.PHONY: setup hooks lint test typecheck format build

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

build:
	uv build
