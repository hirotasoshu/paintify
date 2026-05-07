import json
from pathlib import Path

import numpy as np
import pytest

from paintify.processing.color import rgb_to_lab
from paintify.processing.color_names import NAMED_COLORS, ColorNameMatcher
from paintify.processing.palette import (
    CompactingPaletteBuilder,
    CustomPalette,
    PaletteEntryBuilder,
    PaletteInputError,
)
from paintify.processing.region_table import Region


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


def test_color_name_matcher_uses_vendored_matplotlib_names() -> None:
    assert ("red", "#FF0000") in NAMED_COLORS
    assert ("xkcd:cloudy blue", "#acc2d9") in NAMED_COLORS


def test_color_name_matcher_finds_nearest_name() -> None:
    matcher = ColorNameMatcher((("red", "#ff0000"), ("blue", "#0000ff")))

    assert matcher.closest_name((250, 10, 10)) == "red"


def test_palette_entry_builder_adds_color_names() -> None:
    lab_colors = rgb_to_lab(np.array([[255, 0, 0]], dtype=np.uint8))

    entries = PaletteEntryBuilder().build(lab_colors)

    assert entries[0].name == "red"
