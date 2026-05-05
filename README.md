# paintify

`paintify` is a Python CLI for turning regular images into paint-by-numbers templates.
It takes one input image and generates a printable outline, a colored preview, a numbered paint
palette, and a machine-readable manifest with the exact settings used for the run.

The project is aimed at quick, reproducible template generation rather than professional vector
tracing. It works best for photos or illustrations where the subject has clear shapes and enough
contrast after resizing and smoothing.

## Output Artifacts

Every run writes these files to the output directory:

- `outline.svg`: a printable black-and-white SVG with region boundaries and paint numbers.
- `preview.png`: a colored preview with boundaries and labels, useful for checking the result.
- `palette.json`: the numbered paint palette as JSON.
- `manifest.json`: reproducibility metadata, settings, palette, regions, and label positions.

## Installation

Install the CLI from PyPI with `uv`:

```bash
uv tool install paintify
```

Then run:

```bash
paintify --help
```

You can also run it without a persistent install:

```bash
uvx paintify --help
```

## Install For Development

```bash
uv sync
make hooks
```

`make hooks` requires the project to be inside a Git repository because `prek` installs Git hooks.

## Usage

```bash
paintify input.png --output-dir out --difficulty easy --seed 42
```

`--difficulty` selects a preset (`easy`, `medium`, or `hard`). Presets are a convenient starting
point, and every preset value can be overridden independently:

```bash
paintify input.png \
  --difficulty medium \
  --colors 18 \
  --max-size 900 \
  --min-region-size 24 \
  --smooth-radius 0.5 \
  --starter-palette none \
  --max-regions 500
```

Optional starter palette snapping keeps colors close to a fixed basic paint set:

```bash
paintify input.png --output-dir out --starter-palette basic
```

Use `--starter-palette none` to disable palette snapping explicitly.

For fully manual settings, disable presets with `--no-preset`. In this mode paintify requires all
generation settings to be provided, including an explicit starter palette value (`none` is valid):

```bash
paintify input.png \
  --no-preset \
  --colors 10 \
  --max-size 180 \
  --min-region-size 20 \
  --smooth-radius 0.9 \
  --starter-palette none \
  --max-regions 200
```

## How It Works

`paintify` runs a deterministic image-processing pipeline:

1. The input image is loaded with OpenCV and converted to RGB.
2. The image is resized so its longest side is at most `--max-size`. This controls the working
   resolution and has the biggest impact on speed and region detail.
3. A Gaussian blur with radius `--smooth-radius` is applied. More blur removes noise and produces
   larger, smoother regions; less blur preserves detail.
4. Colors are reduced with deterministic weighted k-means. The algorithm first groups near-identical
   RGB values into small bins, counts how often each binned color appears, and clusters those unique
   colors instead of clustering every pixel. The `--seed` controls the initial cluster centers.
5. Palette colors are compared in Lab color space. Lab distances usually match human color
   perception better than raw RGB distances.
6. If `--starter-palette basic` is used, each generated color is snapped to the nearest color from a
   small built-in starter paint palette.
7. Thin one-pixel strips are cleaned up before region construction.
8. Adjacent pixels with the same palette color are split into connected regions.
9. Small regions below `--min-region-size` are merged into nearby kept regions. If the result still
   has more than `--max-regions`, the smallest regions are removed one at a time and reassigned to
   neighbouring regions with palette-distance tie-breaking.
10. Label positions are placed near the safest inner point of each region by measuring distance from
   the region boundary.
11. The final document is rendered as SVG, PNG, and JSON artifacts.

The same input, settings, and seed should produce the same output.

## Parameters

| Option | Meaning | Practical effect |
| --- | --- | --- |
| `--difficulty` | Preset name: `easy`, `medium`, or `hard`. | Higher difficulty keeps more colors, detail, and regions. |
| `--no-preset` | Disable presets and require explicit settings. | Useful for reproducible manual tuning. |
| `--colors` / `-c` | Maximum number of paint colors before unused colors are compacted away. | More colors preserve detail but make painting harder. |
| `--max-size` | Longest side of the working image in pixels. | Larger values preserve detail but increase runtime and region count. |
| `--min-region-size` | Minimum target region area in working pixels. | Larger values remove tiny islands; smaller values keep fine details. |
| `--smooth-radius` | Gaussian blur radius before color reduction. | Larger values simplify noisy images; `0` disables smoothing. |
| `--starter-palette` | `basic` or `none`. | `basic` limits output colors to a small fixed paint-like palette. |
| `--max-regions` | Maximum number of final numbered regions. | Lower values simplify the template; higher values preserve detail. |
| `--seed` | Deterministic seed for k-means initialization. | Change it to try different color clustering while keeping settings fixed. |
| `--output-dir` / `-o` | Directory for generated artifacts. | Defaults to `paintify-out`. |

## Difficulty Presets

| Difficulty | Colors | Max size | Min region size | Smooth radius | Max regions | Starter palette |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `easy` | 15 | 768 | 40 | 0.6 | 300 | none |
| `medium` | 20 | 1024 | 20 | 0.4 | 700 | none |
| `hard` | 30 | 1280 | 12 | 0.25 | 1200 | none |

Use `easy` for simpler printable templates, `medium` for balanced results, and `hard` when the input
has details you want to preserve.

## JSON Outputs

### `palette.json`

`palette.json` is the simplest output for paint preparation. It contains the final numbered palette
after unused colors are removed and indices are compacted:

```json
[
  {"index": 1, "hex": "#f2d7b6", "rgb": [242, 215, 182]},
  {"index": 2, "hex": "#8b5a2b", "rgb": [139, 90, 43]}
]
```

Use it when you need a shopping/mixing list, want to map numbers to physical paints, or want another
program to read the generated palette.

### `manifest.json`

`manifest.json` is for reproducibility and downstream processing. It records the input path, chosen
difficulty, seed, resolved settings, image size, generated artifact names, full palette, region
metadata, and label positions:

```json
{
  "input": "input.png",
  "difficulty": "easy",
  "seed": 42,
  "settings": {
    "max_colors": 15,
    "max_size": 768,
    "min_region_size": 40,
    "smooth_radius": 0.6,
    "starter_palette": null,
    "max_regions": 300
  },
  "image_size": {"width": 768, "height": 512},
  "artifacts": ["outline.svg", "preview.png", "palette.json", "manifest.json"],
  "palette": [
    {"index": 1, "hex": "#f2d7b6", "rgb": [242, 215, 182]}
  ],
  "regions": [
    {
      "id": 1,
      "palette_index": 1,
      "area": 512,
      "bbox": [10, 8, 45, 33],
      "label": {"x": 27, "y": 19}
    }
  ]
}
```

Important fields:

- `settings`: the fully resolved values after preset application and CLI overrides.
- `image_size`: the working image dimensions after resizing.
- `palette`: the same numbered colors as `palette.json`.
- `regions`: each numbered shape, its palette number, pixel area, bounding box, and label position.
- `bbox`: `[min_x, min_y, max_x, max_y]` in working-image coordinates.

Use `manifest.json` when you want to audit a generated template, compare runs, write custom
renderers, or build UI around the paint-by-numbers data.

## Tuning Tips

- If the result has too many tiny shapes, increase `--min-region-size`, increase `--smooth-radius`,
  lower `--max-regions`, or use `--difficulty easy`.
- If the result loses important details, increase `--max-size`, increase `--colors`, decrease
  `--smooth-radius`, or use `--difficulty hard`.
- If the colors are hard to match with real paints, try `--starter-palette basic`.
- If the segmentation looks odd but the settings are good, change `--seed` to try another color
  clustering.

## Development

```bash
make lint
make test
```

To run individual checks:

```bash
uv run ruff check src tests
uv run flake8 src tests
uv run ty check src tests
uv run pytest
```
