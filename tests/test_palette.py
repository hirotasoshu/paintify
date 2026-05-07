import json
from pathlib import Path

import numpy as np
import pytest

from paintify.processing.color import rgb_to_lab
from paintify.processing.palette import CustomPalette, PaletteEntryBuilder, PaletteInputError


def test_custom_palette_loads_colors_object_hex_values(tmp_path: Path) -> None:
    palette_path = tmp_path / "palette.json"
    palette_path.write_text(
        json.dumps({"colors": ["#d62828", "#277da1"]}),
        encoding="utf-8",
    )

    palette = CustomPalette.load(palette_path)

    assert palette.rgb.tolist() == [[214, 40, 40], [39, 125, 161]]


def test_custom_palette_snapping_uses_nearest_file_color(tmp_path: Path) -> None:
    palette_path = tmp_path / "palette.json"
    palette_path.write_text(
        json.dumps({"colors": ["#d62828", "#277da1"]}),
        encoding="utf-8",
    )
    redish_rgb = np.array([[210, 40, 40]], dtype=np.uint8)
    lab = rgb_to_lab(redish_rgb)

    snapped = CustomPalette.load(palette_path).snap_lab_colors(lab)
    palette = PaletteEntryBuilder().build(snapped)

    assert palette[0].hex == "#d62828"


def test_custom_palette_rejects_invalid_hex(tmp_path: Path) -> None:
    palette_path = tmp_path / "palette.json"
    palette_path.write_text(json.dumps({"colors": ["not-a-color"]}), encoding="utf-8")

    with pytest.raises(PaletteInputError, match="invalid hex color"):
        CustomPalette.load(palette_path)


def test_custom_palette_rejects_output_palette_object_entries(tmp_path: Path) -> None:
    palette_path = tmp_path / "palette.json"
    palette_path.write_text(
        json.dumps({"colors": [{"index": 1, "hex": "#d62828"}]}),
        encoding="utf-8",
    )

    with pytest.raises(PaletteInputError, match="colors must be hex strings"):
        CustomPalette.load(palette_path)


def test_build_palette_deduplicates_entries() -> None:
    white_lab = rgb_to_lab(np.array([[255, 255, 255], [255, 255, 255]], dtype=np.uint8))

    assert len(PaletteEntryBuilder().build(white_lab)) == 1
