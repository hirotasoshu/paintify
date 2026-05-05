from __future__ import annotations

from paintify.pipeline import PaintifyGenerator, Renderer
from paintify.processing import (
    CompactingPaletteBuilder,
    ConnectedComponentRegionProcessor,
    DistanceTransformLabelPlacer,
    KMeansQuantizer,
    OpenCvImageLoader,
)
from paintify.processing.palette import PaletteEntryBuilder
from paintify.rendering.json import ManifestJsonRenderer, PaletteJsonRenderer
from paintify.rendering.png import PngPreviewRenderer
from paintify.rendering.svg import SvgOutlineRenderer
from paintify.rendering.writer import FilesystemArtifactWriter


def create_paintify_generator() -> PaintifyGenerator:
    renderers: list[Renderer] = [
        SvgOutlineRenderer(),
        PngPreviewRenderer(),
        PaletteJsonRenderer(),
        ManifestJsonRenderer(),
    ]
    return PaintifyGenerator(
        image_loader=OpenCvImageLoader(),
        quantizer=KMeansQuantizer(),
        region_processor=ConnectedComponentRegionProcessor(),
        palette_builder=CompactingPaletteBuilder(),
        palette_entry_builder=PaletteEntryBuilder(),
        label_placer=DistanceTransformLabelPlacer(),
        renderers=renderers,
        artifact_writer=FilesystemArtifactWriter(),
    )
