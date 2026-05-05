from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Region:
    id: int
    color_index: int
    area: int
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class LabelPlacement:
    region_id: int
    color_index: int
    x: int
    y: int


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
class PaintByNumbersDocument:
    color_labels: np.ndarray
    region_labels: np.ndarray
    palette: list[PaletteEntry]
    regions: list[Region]
    labels: list[LabelPlacement]

    @property
    def image_size(self) -> tuple[int, int]:
        height, width = self.color_labels.shape
        return width, height


@dataclass(frozen=True)
class OutputArtifact:
    name: str
    payload: str | bytes


@dataclass(frozen=True)
class OutputBundle:
    document: PaintByNumbersDocument
    artifacts: list[OutputArtifact]
