from __future__ import annotations

import numpy as np

from paintify.processing.color import rgb_to_lab
from paintify.processing.palette import PaletteEntryBuilder, StarterPalette


def test_palette_snapping_uses_nearest_starter_color() -> None:
    redish_rgb = np.array([[210, 40, 40]], dtype=np.uint8)
    lab = rgb_to_lab(redish_rgb)

    snapped = StarterPalette.snap_lab_colors(lab, "basic")
    palette = PaletteEntryBuilder().build(snapped)

    assert palette[0].hex == "#d62828"


def test_build_palette_deduplicates_entries() -> None:
    white_lab = rgb_to_lab(np.array([[255, 255, 255], [255, 255, 255]], dtype=np.uint8))

    assert len(PaletteEntryBuilder().build(white_lab)) == 1
