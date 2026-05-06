from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from paintify.processing.color import lab_to_rgb, rgb_to_lab
from paintify.processing.region_table import Region


class PaletteInputError(ValueError):
    pass


@dataclass(frozen=True)
class PaletteEntry:
    index: int
    hex: str
    rgb: tuple[int, int, int]

    @classmethod
    def from_rgb(cls, index: int, rgb: tuple[int, int, int]) -> PaletteEntry:
        return cls(index=index, hex=cls._rgb_to_hex(rgb), rgb=rgb)

    @staticmethod
    def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
        return "#{:02x}{:02x}{:02x}".format(*rgb)


@dataclass(frozen=True)
class CustomPalette:
    rgb: np.ndarray

    hex_color_length = 6

    @classmethod
    def load(cls, path: Path) -> CustomPalette:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except OSError as error:
            raise PaletteInputError(f"could not read palette file: {path}") from error
        except json.JSONDecodeError as error:
            raise PaletteInputError(f"palette file is not valid JSON: {path}") from error
        colors = cls._read_colors(raw)
        return cls(
            rgb=np.array([cls._hex_to_rgb(color) for color in colors], dtype=np.uint8),
        )

    @property
    def color_count(self) -> int:
        return int(self.rgb.shape[0])

    @classmethod
    def _read_colors(cls, raw: Any) -> list[str]:
        if not isinstance(raw, dict) or "colors" not in raw:
            raise PaletteInputError("palette file must contain a colors list")
        colors = raw["colors"]
        if not isinstance(colors, list) or not colors:
            raise PaletteInputError("palette file colors must be a non-empty list")
        if not all(isinstance(color, str) for color in colors):
            raise PaletteInputError("palette file colors must be hex strings")
        return colors

    def snap_lab_colors(self, lab_colors: np.ndarray) -> np.ndarray:
        palette_lab = rgb_to_lab(self.rgb)
        distances = np.linalg.norm(lab_colors[:, None, :] - palette_lab[None, :, :], axis=2)
        return palette_lab[np.argmin(distances, axis=1)]

    @classmethod
    def _hex_to_rgb(cls, value: str) -> tuple[int, int, int]:
        clean = value.removeprefix("#")
        if len(clean) != cls.hex_color_length:
            raise PaletteInputError(f"invalid hex color: {value}")
        try:
            return (int(clean[0:2], 16), int(clean[2:4], 16), int(clean[4:6], 16))
        except ValueError as error:
            raise PaletteInputError(f"invalid hex color: {value}") from error


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
        return lab_to_rgb(lab_colors)


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
