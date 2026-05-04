from __future__ import annotations

import numpy as np
from skimage.color import rgb2lab

from paintify.processing.palette import PaletteEntryBuilder, StarterPalette


def test_palette_snapping_uses_nearest_starter_color() -> None:
    redish_rgb = np.array([[[210, 40, 40]]], dtype=np.uint8) / 255.0
    lab = rgb2lab(redish_rgb).reshape(1, 3)

    snapped = StarterPalette.snap_lab_colors(lab, "basic")
    palette = PaletteEntryBuilder().build(snapped)

    assert palette[0].hex == "#d62828"


def test_build_palette_deduplicates_entries() -> None:
    white_lab = rgb2lab(
        np.array([[[255, 255, 255], [255, 255, 255]]], dtype=np.uint8) / 255.0
    ).reshape(2, 3)

    assert len(PaletteEntryBuilder().build(white_lab)) == 1
