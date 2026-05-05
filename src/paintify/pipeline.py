from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from paintify.config import PaintifyConfig
from paintify.processing.labels import LabelPlacement
from paintify.processing.palette import PaletteEntry
from paintify.processing.region_table import Region
from paintify.rendering.artifacts import OutputArtifact, OutputBundle


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


class ImageLoader(Protocol):
    def load(self, path: Path, max_size: int, smooth_radius: float) -> np.ndarray: ...


class Quantizer(Protocol):
    def quantize(
        self, image: np.ndarray, max_colors: int, seed: int, starter_palette: str | None
    ) -> tuple[np.ndarray, np.ndarray]: ...


class RegionProcessor(Protocol):
    def process(
        self,
        color_labels: np.ndarray,
        lab_palette: np.ndarray,
        min_region_size: int,
        max_regions: int | None,
    ) -> tuple[np.ndarray, np.ndarray, list[Region]]: ...


class PaletteBuilder(Protocol):
    def build(
        self, color_labels: np.ndarray, lab_palette: np.ndarray, regions: list[Region]
    ) -> tuple[np.ndarray, np.ndarray, list[Region]]: ...


class PaletteEntryBuilderProtocol(Protocol):
    def build(self, lab_colors: np.ndarray) -> list[PaletteEntry]: ...


class LabelPlacer(Protocol):
    def place(self, region_labels: np.ndarray, regions: list[Region]) -> list[LabelPlacement]: ...


class Renderer(Protocol):
    def render(
        self, config: PaintifyConfig, document: PaintByNumbersDocument
    ) -> OutputArtifact: ...


class ArtifactWriter(Protocol):
    def write(self, output_dir: Path, bundle: OutputBundle) -> None: ...


@dataclass(frozen=True)
class GenerationResult:
    output_dir: Path
    document: PaintByNumbersDocument

    @property
    def palette(self) -> list[PaletteEntry]:
        return self.document.palette

    @property
    def regions(self) -> list[Region]:
        return self.document.regions

    @property
    def labels(self) -> list[LabelPlacement]:
        return self.document.labels


class PaintifyGenerator:
    def __init__(
        self,
        image_loader: ImageLoader,
        quantizer: Quantizer,
        region_processor: RegionProcessor,
        palette_builder: PaletteBuilder,
        palette_entry_builder: PaletteEntryBuilderProtocol,
        label_placer: LabelPlacer,
        renderers: list[Renderer],
        artifact_writer: ArtifactWriter,
    ) -> None:
        self._image_loader = image_loader
        self._quantizer = quantizer
        self._region_processor = region_processor
        self._palette_builder = palette_builder
        self._palette_entry_builder = palette_entry_builder
        self._label_placer = label_placer
        self._renderers = renderers
        self._artifact_writer = artifact_writer

    def generate(self, config: PaintifyConfig) -> GenerationResult:
        config = config.validated()
        image = self._image_loader.load(config.input_path, config.max_size, config.smooth_radius)
        color_labels, lab_palette = self._quantizer.quantize(
            image, config.max_colors, config.seed, config.starter_palette
        )
        region_labels, color_labels, regions = self._region_processor.process(
            color_labels, lab_palette, config.min_region_size, config.max_regions
        )
        color_labels, lab_palette, regions = self._palette_builder.build(
            color_labels, lab_palette, regions
        )
        palette = self._palette_entry_builder.build(lab_palette)
        document = PaintByNumbersDocument(
            color_labels=color_labels,
            region_labels=region_labels,
            palette=palette,
            regions=regions,
            labels=self._label_placer.place(region_labels, regions),
        )
        bundle = OutputBundle(
            document=document,
            artifacts=[renderer.render(config, document) for renderer in self._renderers],
        )
        self._artifact_writer.write(config.output_dir, bundle)
        return GenerationResult(output_dir=config.output_dir, document=document)
