# Color Names Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add approximate human-readable color names to `palette.json` and `manifest.json` without adding Matplotlib as a dependency.

**Architecture:** Vendor Matplotlib color-name data into a local module, then add a small `ColorNameMatcher` that finds the nearest named color in Lab space. `PaletteEntryBuilder` assigns names when building `PaletteEntry` objects, and JSON renderers serialize the new `name` field.

**Tech Stack:** Python 3.12+, NumPy, existing Lab conversion helpers, uv, pytest, ruff, flake8, ty.

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
git worktree add .worktrees/color-names -b feat/color-names main
```

Expected: Git creates `.worktrees/color-names` on branch `feat/color-names`.

- [ ] **Step 3: Verify baseline tests**

Run from `.worktrees/color-names`:

```bash
uv run pytest
```

Expected: The test suite passes before making changes.

### Task 2: Add Failing Unit Tests

**Files:**
- Modify: `tests/test_palette.py`
- Create later: `src/paintify/processing/color_names.py`

- [ ] **Step 1: Add color-name matcher import**

Update `tests/test_palette.py` imports:

```python
from paintify.processing.color_names import ColorNameMatcher, NAMED_COLORS
```

- [ ] **Step 2: Add tests for vendored names and palette entries**

Append these tests to `tests/test_palette.py`:

```python
def test_color_name_matcher_uses_vendored_matplotlib_names() -> None:
    assert ("red", "#FF0000") in NAMED_COLORS
    assert ("xkcd:cloudy blue", "#acc2d9") in NAMED_COLORS


def test_color_name_matcher_finds_nearest_name() -> None:
    matcher = ColorNameMatcher([("red", "#ff0000"), ("blue", "#0000ff")])

    assert matcher.closest_name((250, 10, 10)) == "red"


def test_palette_entry_builder_adds_color_names() -> None:
    lab_colors = rgb_to_lab(np.array([[255, 0, 0]], dtype=np.uint8))

    [entry] = PaletteEntryBuilder().build(lab_colors)

    assert entry.name == "red"
```

- [ ] **Step 3: Run new tests to verify RED**

Run from `.worktrees/color-names`:

```bash
uv run pytest tests/test_palette.py::test_color_name_matcher_uses_vendored_matplotlib_names tests/test_palette.py::test_color_name_matcher_finds_nearest_name tests/test_palette.py::test_palette_entry_builder_adds_color_names -q
```

Expected: Tests fail because `paintify.processing.color_names` and `PaletteEntry.name` do not exist.

### Task 3: Vendor Matplotlib Color Data

**Files:**
- Create: `src/paintify/processing/color_names.py`

- [ ] **Step 1: Fetch upstream color data to a temporary file**

Run from `.worktrees/color-names`:

```bash
python - <<'PY'
from pathlib import Path
from urllib.request import urlopen

url = "https://raw.githubusercontent.com/matplotlib/matplotlib/f4cc437d1bd72ec72cfc2a0e318c867c6969392f/lib/matplotlib/_color_data.py"
Path("/tmp/opencode/matplotlib_color_data.py").write_text(
    urlopen(url, timeout=60).read().decode("utf-8"),
    encoding="utf-8",
)
PY
```

Expected: `/tmp/opencode/matplotlib_color_data.py` exists.

- [ ] **Step 2: Generate local color-name module**

Run from `.worktrees/color-names`:

```bash
python - <<'PY'
from pathlib import Path

namespace: dict[str, object] = {}
source = Path("/tmp/opencode/matplotlib_color_data.py").read_text(encoding="utf-8")
exec(source, namespace)
css4 = namespace["CSS4_COLORS"]
xkcd = namespace["XKCD_COLORS"]
colors = list(css4.items()) + list(xkcd.items())
lines = [
    '"""Vendored named color data and nearest-name matching."""',
    '',
    '# Data derived from Matplotlib:',
    '# https://github.com/matplotlib/matplotlib/blob/f4cc437d1bd72ec72cfc2a0e318c867c6969392f/lib/matplotlib/_color_data.py',
    '# CSS4 colors are from https://drafts.csswg.org/css-color-4/#named-colors',
    '# XKCD colors are from https://xkcd.com/color/rgb/ and are CC0:',
    '# https://creativecommons.org/publicdomain/zero/1.0/',
    '',
    'import numpy as np',
    '',
    'from paintify.processing.color import rgb_to_lab',
    '',
    'NAMED_COLORS: tuple[tuple[str, str], ...] = (',
]
for name, hex_value in colors:
    lines.append(f'    ({name!r}, {hex_value!r}),')
lines.extend([
    ')',
    '',
    '',
    'class ColorNameMatcher:',
    '    def __init__(self, named_colors: tuple[tuple[str, str], ...] = NAMED_COLORS) -> None:',
    '        self._names = [name for name, _ in named_colors]',
    '        self._lab = rgb_to_lab(np.array([self._hex_to_rgb(value) for _, value in named_colors], dtype=np.uint8))',
    '',
    '    def closest_name(self, rgb: tuple[int, int, int]) -> str:',
    '        lab = rgb_to_lab(np.array([rgb], dtype=np.uint8))',
    '        distances = np.linalg.norm(self._lab - lab, axis=1)',
    '        return self._names[int(np.argmin(distances))]',
    '',
    '    @staticmethod',
    '    def _hex_to_rgb(value: str) -> tuple[int, int, int]:',
    '        clean = value.removeprefix("#")',
    '        return (int(clean[0:2], 16), int(clean[2:4], 16), int(clean[4:6], 16))',
    '',
])
Path("src/paintify/processing/color_names.py").write_text("\n".join(lines), encoding="utf-8")
PY
```

Expected: `src/paintify/processing/color_names.py` contains attribution comments, `NAMED_COLORS`, and `ColorNameMatcher`.

### Task 4: Add Names To Palette Entries And JSON

**Files:**
- Modify: `src/paintify/processing/palette.py`
- Modify: `src/paintify/rendering/json.py`

- [ ] **Step 1: Add `name` to `PaletteEntry`**

Update `PaletteEntry` and `PaletteEntry.from_rgb` in `src/paintify/processing/palette.py`:

```python
@dataclass(frozen=True)
class PaletteEntry:
    index: int
    hex: str
    rgb: tuple[int, int, int]
    name: str

    @classmethod
    def from_rgb(cls, index: int, rgb: tuple[int, int, int], name: str) -> "PaletteEntry":
        return cls(index=index, hex=cls._rgb_to_hex(rgb), rgb=rgb, name=name)
```

- [ ] **Step 2: Use `ColorNameMatcher` in `PaletteEntryBuilder`**

Import and initialize the matcher:

```python
from paintify.processing.color_names import ColorNameMatcher
```

Update `PaletteEntryBuilder`:

```python
class PaletteEntryBuilder:
    def __init__(self, color_name_matcher: ColorNameMatcher | None = None) -> None:
        self._color_name_matcher = color_name_matcher or ColorNameMatcher()

    def build(self, lab_colors: np.ndarray) -> list[PaletteEntry]:
        rgb_values = self._lab_to_uint8_rgb(lab_colors)
        entries: list[PaletteEntry] = []
        seen: set[tuple[int, int, int]] = set()
        for rgb_array in rgb_values:
            rgb = (int(rgb_array[0]), int(rgb_array[1]), int(rgb_array[2]))
            if rgb in seen:
                continue
            seen.add(rgb)
            entries.append(
                PaletteEntry.from_rgb(
                    index=len(entries) + 1,
                    rgb=rgb,
                    name=self._color_name_matcher.closest_name(rgb),
                )
            )
        return entries
```

- [ ] **Step 3: Serialize `name` in JSON renderers**

Update both palette serialization locations in `src/paintify/rendering/json.py`:

```python
{"index": entry.index, "hex": entry.hex, "rgb": list(entry.rgb), "name": entry.name}
```

- [ ] **Step 4: Run unit tests to verify GREEN**

Run from `.worktrees/color-names`:

```bash
uv run pytest tests/test_palette.py::test_color_name_matcher_uses_vendored_matplotlib_names tests/test_palette.py::test_color_name_matcher_finds_nearest_name tests/test_palette.py::test_palette_entry_builder_adds_color_names -q
```

Expected: The tests pass.

### Task 5: Add JSON Integration Coverage And Docs

**Files:**
- Modify: `tests/test_integration.py`
- Modify: `README.md`

- [ ] **Step 1: Assert JSON outputs include names**

In `tests/test_integration.py::test_palette_output_is_compacted_after_region_merge`, add:

```python
    assert isinstance(palette[0]["name"], str)
    assert palette[0]["name"]
    assert isinstance(manifest["palette"][0]["name"], str)
    assert manifest["palette"][0]["name"]
```

- [ ] **Step 2: Update README JSON examples**

In `README.md`, update palette examples so entries include `name`, for example:

```json
{"index": 1, "hex": "#f2d7b6", "rgb": [242, 215, 182], "name": "xkcd:light peach"}
```

Also add one sentence near the palette JSON section:

```markdown
`name` is the nearest CSS4 or XKCD color name from vendored Matplotlib color data; it is a descriptive approximation, not a paint catalog match.
```

- [ ] **Step 3: Run integration-focused tests**

Run from `.worktrees/color-names`:

```bash
uv run pytest tests/test_integration.py::test_palette_output_is_compacted_after_region_merge -q
```

Expected: The test passes.

### Task 6: Verify, Commit, Push, PR

**Files:**
- All modified files from previous tasks

- [ ] **Step 1: Run full checks**

Run from `.worktrees/color-names`:

```bash
uv run ruff check src tests
uv run flake8 src tests
uv run ty check src tests
uv run pytest
```

Expected: All commands pass.

- [ ] **Step 2: Commit**

Run from `.worktrees/color-names`:

```bash
git add README.md src/paintify/processing/color_names.py src/paintify/processing/palette.py src/paintify/rendering/json.py tests/test_palette.py tests/test_integration.py
git commit -m "feat: add palette color names"
```

Expected: A single feature commit is created.

- [ ] **Step 3: Push and create PR**

Run from `.worktrees/color-names`:

```bash
git push -u origin feat/color-names
gh pr create --title "Add palette color names" --body "$(cat <<'EOF'
## Summary
- Add approximate human-readable names to palette entries.
- Vendor CSS4 and XKCD color-name data derived from Matplotlib without adding Matplotlib as a dependency.
- Include names in `palette.json`, `manifest.json`, and README examples.

## Test Plan
- `uv run ruff check src tests`
- `uv run flake8 src tests`
- `uv run ty check src tests`
- `uv run pytest`
EOF
)"
```

Expected: GitHub returns a PR URL.
