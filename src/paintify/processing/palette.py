from __future__ import annotations

from dataclasses import replace
from typing import cast

import numpy as np
from skimage.color import lab2rgb, rgb2lab

from paintify.models import PaletteEntry, Region


class StarterPalette:
    PALETTES: dict[str, list[str]] = {
        "basic": [
            "#ffffff",
            "#000000",
            "#d62828",
            "#f77f00",
            "#fcbf49",
            "#2a9d8f",
            "#277da1",
            "#5e548e",
            "#8d5524",
            "#6a994e",
        ]
    }

    @classmethod
    def snap_lab_colors(cls, lab_colors: np.ndarray, starter_name: str | None) -> np.ndarray:
        if starter_name is None:
            return lab_colors
        if starter_name not in cls.PALETTES:
            raise ValueError(f"unknown starter palette: {starter_name}")
        starter_rgb = np.array(
            [cls._hex_to_rgb(value) for value in cls.PALETTES[starter_name]], dtype=np.uint8
        )
        starter_lab = rgb2lab(starter_rgb.reshape(1, -1, 3) / 255.0).reshape(-1, 3)
        distances = np.linalg.norm(lab_colors[:, None, :] - starter_lab[None, :, :], axis=2)
        snapped = starter_lab[np.argmin(distances, axis=1)]
        return cast(np.ndarray, snapped)

    @staticmethod
    def _hex_to_rgb(value: str) -> tuple[int, int, int]:
        clean = value.removeprefix("#")
        if len(clean) != 6:
            raise ValueError(f"invalid hex color: {value}")
        return (int(clean[0:2], 16), int(clean[2:4], 16), int(clean[4:6], 16))


class PaletteEntryBuilder:
    def build(self, lab_colors: np.ndarray) -> list[PaletteEntry]:
        rgb_values = self._lab_to_uint8_rgb(lab_colors)
        entries: list[PaletteEntry] = []
        seen: set[tuple[int, int, int]] = set()
        for rgb_array in rgb_values:
            rgb = (int(rgb_array[0]), int(rgb_array[1]), int(rgb_array[2]))
            if rgb in seen:
                continue
            seen.add(rgb)
            entries.append(PaletteEntry.from_rgb(index=len(entries) + 1, rgb=rgb))
        return entries

    @staticmethod
    def _lab_to_uint8_rgb(lab_colors: np.ndarray) -> np.ndarray:
        rgb = lab2rgb(lab_colors.reshape(1, -1, 3)).reshape(-1, 3)
        clipped = np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8)
        return cast(np.ndarray, clipped)


class CompactingPaletteBuilder:
    def build(
        self,
        color_labels: np.ndarray,
        lab_palette: np.ndarray,
        regions: list[Region],
    ) -> tuple[np.ndarray, np.ndarray, list[Region]]:
        used_indices = sorted(int(value) for value in np.unique(color_labels))
        index_map = {old_index: new_index for new_index, old_index in enumerate(used_indices)}
        compact_labels = np.zeros_like(color_labels, dtype=np.int32)
        for old_index, new_index in index_map.items():
            compact_labels[color_labels == old_index] = new_index
        compact_regions = [
            replace(region, color_index=index_map[region.color_index]) for region in regions
        ]
        compact_palette = lab_palette[used_indices]
        return compact_labels, compact_palette, compact_regions
