# Debug Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in CLI diagnostics with `--verbose` console output and `--debug-dir` structured debug stats.

**Architecture:** Keep this first pass at the CLI boundary. The pipeline already returns enough data for useful diagnostics, so `main.py` will print verbose details after generation and write `debug-stats.json` only when `--debug-dir` is provided.

**Tech Stack:** Python 3.12+, Typer, Rich console, JSON, uv, pytest, ruff, flake8, ty.

---

### Task 1: Add CLI tests

**Files:**
- Modify: `tests/test_cli.py`

- [ ] Add tests proving `--verbose` prints diagnostic fields and `--debug-dir` writes `debug-stats.json`.
- [ ] Run the new tests and verify they fail before implementation.

### Task 2: Implement CLI diagnostics

**Files:**
- Modify: `src/paintify/cli/main.py`

- [ ] Add `verbose: bool = typer.Option(False, "--verbose", help="Print diagnostic details.")`.
- [ ] Add `debug_dir: Path | None = typer.Option(None, "--debug-dir", help="Write debug diagnostics to this directory.")`.
- [ ] After successful generation, print concise verbose details only when `verbose` is true.
- [ ] When `debug_dir` is set, create it and write `debug-stats.json` with input, output_dir, image_size, settings, artifacts, color_count, and region_count.

### Task 3: Verify, commit, PR

**Files:**
- Modified files from Tasks 1 and 2

- [ ] Run `uv run ruff check src tests`, `uv run flake8 src tests`, `uv run ty check src tests`, and `uv run pytest`.
- [ ] Commit with `feat: add debug diagnostics`.
- [ ] Push `feat/debug-logging` and open PR.
