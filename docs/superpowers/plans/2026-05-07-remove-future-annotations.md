# Remove Future Annotations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove unnecessary `from __future__ import annotations` imports from the Python 3.12+ codebase.

**Architecture:** This is a mechanical cleanup with no behavior changes. Remove the future import line from every source and test module, then run the existing lint, type, and test checks to prove the cleanup did not change runtime behavior.

**Tech Stack:** Python 3.12+, uv, ruff, flake8, ty, pytest.

---

### Task 1: Create Cleanup Branch

**Files:**
- No file changes in this task

- [ ] **Step 1: Ensure `main` is clean and current**

Run:

```bash
git status --short --branch
git push origin main
```

Expected: `main` is clean, and local commits containing the roadmap spec are pushed.

- [ ] **Step 2: Create an isolated worktree branch**

Run:

```bash
git worktree add .worktrees/remove-future-annotations -b chore/remove-future-annotations main
```

Expected: Git creates `.worktrees/remove-future-annotations` on branch `chore/remove-future-annotations`.

- [ ] **Step 3: Verify baseline checks in the worktree**

Run from `.worktrees/remove-future-annotations`:

```bash
uv run pytest
```

Expected: The test suite passes before making changes.

### Task 2: Remove Future Imports

**Files:**
- Modify: every `src/**/*.py` and `tests/**/*.py` file containing `from __future__ import annotations`

- [ ] **Step 1: List all future annotation imports**

Run from `.worktrees/remove-future-annotations`:

```bash
rg "^from __future__ import annotations$" src tests
```

Expected: The command prints all files that still contain the import.

- [ ] **Step 2: Remove the import from each matching file**

For every file printed by the previous step, remove this exact line:

```python
from __future__ import annotations
```

If the removal leaves two blank lines at the top of a file, collapse them so each file starts cleanly
with the first real import or module code.

- [ ] **Step 3: Verify no imports remain**

Run from `.worktrees/remove-future-annotations`:

```bash
rg "^from __future__ import annotations$" src tests
```

Expected: No matches.

### Task 3: Verify And Commit

**Files:**
- Modified files from Task 2

- [ ] **Step 1: Run lint, type, and test checks**

Run from `.worktrees/remove-future-annotations`:

```bash
uv run ruff check src tests
uv run flake8 src tests
uv run ty check src tests
uv run pytest
```

Expected: All commands pass.

- [ ] **Step 2: Review the diff**

Run from `.worktrees/remove-future-annotations`:

```bash
git diff
```

Expected: The diff only removes `from __future__ import annotations` lines and any extra blank lines
left behind by those removals.

- [ ] **Step 3: Commit the cleanup**

Run from `.worktrees/remove-future-annotations`:

```bash
git add src tests
git commit -m "chore: remove future annotation imports"
```

Expected: A single cleanup commit is created.

### Task 4: Push And Open PR

**Files:**
- No additional file changes

- [ ] **Step 1: Push the branch**

Run from `.worktrees/remove-future-annotations`:

```bash
git push -u origin chore/remove-future-annotations
```

Expected: Branch is pushed to GitHub.

- [ ] **Step 2: Create the pull request**

Run from `.worktrees/remove-future-annotations`:

```bash
gh pr create --title "Remove future annotation imports" --body "$(cat <<'EOF'
## Summary
- Remove unnecessary `from __future__ import annotations` imports from Python 3.12+ source and test files.
- Keep this as a mechanical cleanup before behavior-changing feature PRs.

## Test Plan
- `uv run ruff check src tests`
- `uv run flake8 src tests`
- `uv run ty check src tests`
- `uv run pytest`
EOF
)"
```

Expected: GitHub returns a PR URL.
