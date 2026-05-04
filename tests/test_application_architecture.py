from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
from skimage.color import rgb2lab

from paintify.config import PaintifyConfig
from paintify.models import (
    LabelPlacement,
    OutputArtifact,
    OutputBundle,
    PaintByNumbersDocument,
    PaletteEntry,
    Region,
)
from paintify.pipeline import PaintifyGenerator
from paintify.rendering import (
    FilesystemArtifactWriter,
    ManifestJsonRenderer,
    PaletteJsonRenderer,
    PngPreviewRenderer,
    SvgOutlineRenderer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_generator_orchestrates_collaborators_around_render_document(tmp_path: Path) -> None:
    class FakeImageLoader:
        def load(self, path: Path, max_size: int, smooth_radius: float) -> np.ndarray:
            assert path == tmp_path / "input.png"
            assert max_size == 8
            assert smooth_radius == 0
            return np.zeros((1, 1, 3), dtype=np.uint8)

    class FakeQuantizer:
        def quantize(
            self,
            image: np.ndarray,
            max_colors: int,
            seed: int,
            starter_palette: str | None,
        ) -> tuple[np.ndarray, np.ndarray]:
            assert image.shape == (1, 1, 3)
            assert (max_colors, seed, starter_palette) == (2, 7, None)
            lab_palette = rgb2lab(np.array([[[255, 255, 255]]], dtype=np.uint8) / 255.0).reshape(
                1, 3
            )
            return np.zeros((1, 1), dtype=np.int32), lab_palette

    class FakeRegionProcessor:
        def process(
            self,
            color_labels: np.ndarray,
            min_region_size: int,
            max_regions: int | None,
        ) -> tuple[np.ndarray, np.ndarray, list[Region]]:
            assert (min_region_size, max_regions) == (1, 3)
            return np.ones((1, 1), dtype=np.int32), color_labels, [Region(1, 0, 1, (0, 0, 1, 1))]

    class FakePaletteBuilder:
        def build(
            self,
            color_labels: np.ndarray,
            lab_palette: np.ndarray,
            regions: list[Region],
        ) -> tuple[np.ndarray, np.ndarray, list[Region]]:
            return color_labels, lab_palette, regions

    class FakeLabelPlacer:
        def place(self, region_labels: np.ndarray, regions: list[Region]) -> list[LabelPlacement]:
            assert int(region_labels[0, 0]) == 1
            return [LabelPlacement(regions[0].id, regions[0].color_index, 0, 0)]

    class CapturingRenderer:
        document: PaintByNumbersDocument | None = None

        def render(
            self, config: PaintifyConfig, document: PaintByNumbersDocument
        ) -> OutputArtifact:
            self.document = document
            return OutputArtifact("custom.txt", f"{document.image_size}:{len(document.palette)}")

    class CapturingWriter:
        output_dir: Path | None = None
        bundle: OutputBundle | None = None

        def write(self, output_dir: Path, bundle: OutputBundle) -> None:
            self.output_dir = output_dir
            self.bundle = bundle

    renderer = CapturingRenderer()
    writer = CapturingWriter()
    config = PaintifyConfig(
        input_path=tmp_path / "input.png",
        output_dir=tmp_path / "out",
        max_colors=2,
        max_size=8,
        min_region_size=1,
        smooth_radius=0,
        starter_palette=None,
        max_regions=3,
        seed=7,
    )

    result = PaintifyGenerator(
        image_loader=FakeImageLoader(),
        quantizer=FakeQuantizer(),
        region_processor=FakeRegionProcessor(),
        palette_builder=FakePaletteBuilder(),
        label_placer=FakeLabelPlacer(),
        renderers=[renderer],
        artifact_writer=writer,
    ).generate(config)

    assert renderer.document is result.document
    assert writer.output_dir == tmp_path / "out"
    assert writer.bundle == OutputBundle(
        result.document,
        [OutputArtifact("custom.txt", "(1, 1):1")],
    )
    assert result.palette == [PaletteEntry(1, "#ffffff", (255, 255, 255))]
    assert result.regions == [Region(1, 0, 1, (0, 0, 1, 1))]
    assert result.labels == [LabelPlacement(1, 0, 0, 0)]


def test_renderers_return_named_payloads_and_writer_persists_them(tmp_path: Path) -> None:
    config = PaintifyConfig(
        input_path=tmp_path / "input.png",
        output_dir=tmp_path / "out",
        max_colors=2,
        max_size=8,
        min_region_size=1,
        smooth_radius=0,
        starter_palette=None,
        max_regions=1,
        seed=0,
    )
    document = PaintByNumbersDocument(
        color_labels=np.zeros((1, 1), dtype=np.int32),
        region_labels=np.ones((1, 1), dtype=np.int32),
        palette=[PaletteEntry(1, "#ffffff", (255, 255, 255))],
        regions=[Region(1, 0, 1, (0, 0, 1, 1))],
        labels=[LabelPlacement(1, 0, 0, 0)],
    )

    artifacts = [
        SvgOutlineRenderer().render(config, document),
        PngPreviewRenderer().render(config, document),
        PaletteJsonRenderer().render(config, document),
        ManifestJsonRenderer().render(config, document),
    ]
    FilesystemArtifactWriter().write(config.output_dir, OutputBundle(document, artifacts))

    assert [artifact.name for artifact in artifacts] == [
        "outline.svg",
        "preview.png",
        "palette.json",
        "manifest.json",
    ]
    assert (config.output_dir / "outline.svg").read_text(encoding="utf-8").startswith("<svg")
    assert (config.output_dir / "preview.png").read_bytes().startswith(b"\x89PNG")
    palette = json.loads((config.output_dir / "palette.json").read_text(encoding="utf-8"))
    manifest = json.loads((config.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert palette == [{"index": 1, "hex": "#ffffff", "rgb": [255, 255, 255]}]
    assert manifest["artifacts"] == ["outline.svg", "preview.png", "palette.json", "manifest.json"]


def test_architecture_uses_simple_models_processing_pipeline_boundaries() -> None:
    from paintify.models import PaintByNumbersDocument as ModelDocument
    from paintify.pipeline import PaintifyGenerator as PipelineGenerator
    from paintify.processing.regions import ConnectedComponentRegionProcessor

    assert PipelineGenerator is PaintifyGenerator
    assert ModelDocument is PaintByNumbersDocument
    assert ConnectedComponentRegionProcessor.__module__ == "paintify.processing.regions"


def test_no_clean_architecture_packages_remain() -> None:
    assert not (PROJECT_ROOT / "src" / "paintify" / "application").exists()
    assert not (PROJECT_ROOT / "src" / "paintify" / "domain").exists()


def test_rendering_does_not_import_processing_or_pipeline_modules() -> None:
    rendering_files = (PROJECT_ROOT / "src" / "paintify" / "rendering").glob("*.py")
    forbidden_prefixes = ("paintify.processing", "paintify.pipeline", "paintify.painting")

    for path in rendering_files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import | ast.ImportFrom)]
        for node in imports:
            if isinstance(node, ast.ImportFrom):
                module_names = [node.module or ""]
            else:
                module_names = [alias.name for alias in node.names]
            assert not any(
                module_name.startswith(forbidden_prefixes) for module_name in module_names
            ), f"{path.relative_to(PROJECT_ROOT)} imports {module_names}"
