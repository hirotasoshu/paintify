# Feature Roadmap Design

## Context

`paintify` is a Python 3.12+ CLI that turns an input image into paint-by-numbers artifacts:
`outline.svg`, `preview.png`, `palette.json`, and `manifest.json`. The current pipeline supports
difficulty presets, manual overrides, deterministic seeds, custom palette JSON files, region
reduction, label placement, and reproducibility metadata.

This roadmap will be implemented as separate pull requests, merged one at a time into `main`, then
released together in a new version.

## Branch And PR Sequence

1. `chore/remove-future-annotations`
2. `feat/stable-color-numbering`
3. `feat/color-names`
4. `feat/debug-logging`
5. Release branch or direct release commit after all feature PRs are merged

Each PR starts from the latest `main`, includes focused tests, and is merged before the next feature
branch starts. The release happens only after all four implementation PRs are merged and verified.

## PR 1: Remove Future Annotations

Remove `from __future__ import annotations` from `src` and `tests`. The project requires Python
`>=3.12`, so postponed annotation evaluation is not needed for the current type syntax used in the
codebase. This is a mechanical cleanup PR to reduce header noise before behavior-changing work.

Verification:

- `uv run ruff check src tests`
- `uv run flake8 src tests`
- `uv run ty check src tests`
- `uv run pytest`

## PR 2: Stable Color Numbering

Make palette numbering deterministic and human-readable by sorting final colors visually after unused
colors are removed. The selected order is `hue -> lightness -> saturation`.

The implementation should remap all color labels, regions, palette entries, SVG labels, PNG labels,
`palette.json`, and `manifest.json` consistently. This feature intentionally changes generated output
numbering, but does not change segmentation, palette color values, or region geometry.

Expected behavior:

- The same final set of colors always receives the same visual order.
- Labels in rendered artifacts match the renumbered palette.
- Manifest region `palette_index` values match the renumbered palette.
- Sorting is based on color appearance, not internal k-means cluster order or area.

## PR 3: Color Names

Add human-readable color names to palette outputs. Each palette entry gains a `name` field in
`palette.json` and `manifest.json`.

Use vendored color-name data derived from Matplotlib's `lib/matplotlib/_color_data.py` at commit
`f4cc437d1bd72ec72cfc2a0e318c867c6969392f`, without adding Matplotlib as a dependency. The vendored
module must include an attribution comment with the source URL and the relevant license/source notes
from the upstream file. The first implementation should vendor CSS4 named colors and XKCD survey
colors from that file, then normalize display names for paintify output.

Name matching should be deterministic and local. A nearest-color match in Lab space is preferred
because the rest of the project already uses Lab distances for perceptual palette operations.

Expected behavior:

- `palette.json` entries include `name`.
- `manifest.json` palette entries include the same `name`.
- Color names are approximate labels, not physical paint catalog matches.
- No new runtime dependency is introduced.

## PR 4: Debug Logging

Add explicit diagnostic modes without changing normal CLI output.

`--verbose` prints useful pipeline progress and summary information to the console, including input
size, working size, palette size before and after compaction, region counts before and after
reduction, output paths, and elapsed timings where practical.

`--debug-dir DIR` writes opt-in debug artifacts. The exact artifact set can stay minimal in the first
iteration, but should focus on information that helps diagnose pipeline quality, such as quantized
preview images, region maps, and structured stats. Debug artifacts must not be written unless the
user explicitly passes `--debug-dir`.

Expected behavior:

- Default output remains concise.
- `--verbose` adds console diagnostics only.
- `--debug-dir` writes diagnostics outside the normal artifact set.
- Debugging support is implemented through clear pipeline interfaces rather than ad-hoc global state.

## Release

After all PRs are merged:

1. Bump the version with `uv version`.
2. Verify build and tests.
3. Commit and push the release commit.
4. Create a GitHub release with `gh release create`.
5. Publish to PyPI if that remains part of the release process.

Release notes should mention stable visual color numbering, palette color names, verbose/debug
diagnostics, and the removal of unnecessary future annotation imports.
