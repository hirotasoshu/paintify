# Stable Color Numbering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make final palette numbering stable and human-readable by sorting compacted colors by `hue -> lightness -> saturation`.

**Architecture:** Add visual color ordering inside `CompactingPaletteBuilder`, after unused palette entries are removed and before renderers build output artifacts. The builder will reorder `lab_palette`, remap `color_labels`, and update each `Region.color_index`; existing renderers then continue using `color_index + 1` without knowing about the sorting step.

**Tech Stack:** Python 3.12+, NumPy, OpenCV color conversion helpers, uv, pytest, ruff, flake8, ty.

---

### Task 1: Create Feature Branch

**Files:**
- No file changes in this task

- [ ] **Step 1: Ensure `main` is current**

Run:

```bash
git status --short --branch
git pull --ff-only origin main
```

Expected: `main` is clean and up to date.

- [ ] **Step 2: Create an isolated worktree branch**

Run:

```bash
git worktree add .worktrees/stable-color-numbering -b feat/stable-color-numbering main
```

Expected: Git creates `.worktrees/stable-color-numbering` on branch `feat/stable-color-numbering`.

- [ ] **Step 3: Verify baseline tests in the worktree**

Run from `.worktrees/stable-color-numbering`:

```bash
uv run pytest
```

Expected: The test suite passes before making changes.

### Task 2: Add Failing Palette Ordering Test

**Files:**
- Modify: `tests/test_palette.py`

- [ ] **Step 1: Add imports for the builder and region type**

Update the imports in `tests/test_palette.py` to include `CompactingPaletteBuilder` and `Region`:

```python
from paintify.processing.palette import (
    CompactingPaletteBuilder,
    CustomPalette,
    PaletteEntryBuilder,
    PaletteInputError,
)
from paintify.processing.region_table import Region
```

- [ ] **Step 2: Add the failing behavior test**

Append this test to `tests/test_palette.py`:

```python
def test_compacting_palette_builder_sorts_colors_by_visual_order() -> None:
    lab_palette = rgb_to_lab(
        np.array(
            [
                [0, 0, 255],
                [255, 0, 0],
                [0, 255, 0],
            ],
            dtype=np.uint8,
        )
    )
    color_labels = np.array([[0, 1, 2]], dtype=np.int32)
    regions = [
        Region(id=1, color_index=0, area=1, bbox=(0, 0, 1, 1)),
        Region(id=2, color_index=1, area=1, bbox=(1, 0, 2, 1)),
        Region(id=3, color_index=2, area=1, bbox=(2, 0, 3, 1)),
    ]

    compact_labels, compact_palette, compact_regions = CompactingPaletteBuilder().build(
        color_labels, lab_palette, regions
    )
    entries = PaletteEntryBuilder().build(compact_palette)

    assert [entry.hex for entry in entries] == ["#ff0000", "#00ff00", "#0000ff"]
    assert compact_labels.tolist() == [[2, 0, 1]]
    assert [region.color_index for region in compact_regions] == [2, 0, 1]
```

- [ ] **Step 3: Run the new test to verify it fails**

Run from `.worktrees/stable-color-numbering`:

```bash
uv run pytest tests/test_palette.py::test_compacting_palette_builder_sorts_colors_by_visual_order -q
```

Expected: The test fails because current compaction preserves internal palette index order and returns `#0000ff`, `#ff0000`, `#00ff00`.

### Task 3: Implement Visual Palette Ordering

**Files:**
- Modify: `src/paintify/processing/palette.py`

- [ ] **Step 1: Add standard-library color conversion import**

At the top of `src/paintify/processing/palette.py`, add:

```python
import colorsys
```

- [ ] **Step 2: Remap compacted palette by visual order**

Update `CompactingPaletteBuilder.build` so the compacted palette is sorted by visual key and labels/regions are remapped consistently:

```python
class CompactingPaletteBuilder:
    def build(
        self,
        color_labels: np.ndarray,
        lab_palette: np.ndarray,
        regions: list[Region],
    ) -> tuple[np.ndarray, np.ndarray, list[Region]]:
        used_indices = sorted(int(value) for value in np.unique(color_labels))
        compact_index_map = {
            old_index: new_index for new_index, old_index in enumerate(used_indices)
        }
        compact_labels = np.zeros_like(color_labels, dtype=np.int32)
        for old_index, new_index in compact_index_map.items():
            compact_labels[color_labels == old_index] = new_index
        compact_regions = [
            replace(region, color_index=compact_index_map[region.color_index])
            for region in regions
        ]
        compact_palette = lab_palette[used_indices]
        visual_order = self._visual_order(compact_palette)
        visual_index_map = {
            old_index: new_index for new_index, old_index in enumerate(visual_order)
        }
        visual_labels = np.zeros_like(compact_labels, dtype=np.int32)
        for old_index, new_index in visual_index_map.items():
            visual_labels[compact_labels == old_index] = new_index
        visual_regions = [
            replace(region, color_index=visual_index_map[region.color_index])
            for region in compact_regions
        ]
        return visual_labels, compact_palette[visual_order], visual_regions
```

- [ ] **Step 3: Add helper methods for visual sorting**

Add these methods to `CompactingPaletteBuilder`:

```python
    @classmethod
    def _visual_order(cls, lab_palette: np.ndarray) -> list[int]:
        rgb_palette = lab_to_rgb(lab_palette)
        return sorted(
            range(len(rgb_palette)),
            key=lambda index: cls._visual_sort_key(rgb_palette[index]),
        )

    @staticmethod
    def _visual_sort_key(rgb: np.ndarray) -> tuple[float, float, float]:
        red = int(rgb[0]) / 255.0
        green = int(rgb[1]) / 255.0
        blue = int(rgb[2]) / 255.0
        hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)
        return hue, lightness, saturation
```

- [ ] **Step 4: Run the new test to verify it passes**

Run from `.worktrees/stable-color-numbering`:

```bash
uv run pytest tests/test_palette.py::test_compacting_palette_builder_sorts_colors_by_visual_order -q
```

Expected: The test passes.

### Task 4: Verify Integration And Commit

**Files:**
- Modified files from Tasks 2 and 3

- [ ] **Step 1: Run full checks**

Run from `.worktrees/stable-color-numbering`:

```bash
uv run ruff check src tests
uv run flake8 src tests
uv run ty check src tests
uv run pytest
```

Expected: All commands pass.

- [ ] **Step 2: Review diff**

Run from `.worktrees/stable-color-numbering`:

```bash
git diff
```

Expected: Diff is limited to palette ordering implementation and focused tests.

- [ ] **Step 3: Commit**

Run from `.worktrees/stable-color-numbering`:

```bash
git add src/paintify/processing/palette.py tests/test_palette.py
git commit -m "feat: sort palette numbers by color"
```

Expected: A single feature commit is created.

### Task 5: Push And Open PR

**Files:**
- No additional file changes

- [ ] **Step 1: Push branch**

Run from `.worktrees/stable-color-numbering`:

```bash
git push -u origin feat/stable-color-numbering
```

Expected: Branch is pushed to GitHub.

- [ ] **Step 2: Create PR**

Run from `.worktrees/stable-color-numbering`:

```bash
gh pr create --title "Sort palette numbers by color" --body "$(cat <<'EOF'
## Summary
- Sort final compacted palette colors by hue, lightness, and saturation.
- Remap color labels and region palette indices so rendered labels match the sorted palette.

## Test Plan
- `uv run ruff check src tests`
- `uv run flake8 src tests`
- `uv run ty check src tests`
- `uv run pytest`
EOF
)"
```

Expected: GitHub returns a PR URL.
