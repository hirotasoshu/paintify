from __future__ import annotations

import json
from typing import Any

from paintify.config import PaintifyConfig
from paintify.pipeline import PaintByNumbersDocument
from paintify.processing.labels import LabelPlacement
from paintify.processing.palette import PaletteEntry
from paintify.processing.region_table import Region
from paintify.rendering.artifacts import OutputArtifact


class PaletteJsonRenderer:
    artifact_name = "palette.json"

    def render(self, config: PaintifyConfig, document: PaintByNumbersDocument) -> OutputArtifact:
        del config
        return OutputArtifact(self.artifact_name, self._serialize_palette(document.palette))

    def _serialize_palette(self, palette: list[PaletteEntry]) -> str:
        data = [
            {"index": entry.index, "hex": entry.hex, "rgb": list(entry.rgb)} for entry in palette
        ]
        return json.dumps(data, indent=2) + "\n"


class ManifestJsonRenderer:
    artifact_name = "manifest.json"

    def render(self, config: PaintifyConfig, document: PaintByNumbersDocument) -> OutputArtifact:
        return OutputArtifact(self.artifact_name, self._serialize_manifest(config, document))

    def _serialize_manifest(self, config: PaintifyConfig, document: PaintByNumbersDocument) -> str:
        return (
            json.dumps(
                self._build_payload(
                    config,
                    document.image_size,
                    document.palette,
                    document.regions,
                    document.labels,
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    def _build_payload(
        self,
        config: PaintifyConfig,
        image_size: tuple[int, int],
        palette: list[PaletteEntry],
        regions: list[Region],
        labels: list[LabelPlacement],
    ) -> dict[str, Any]:
        label_map = {label.region_id: label for label in labels}
        return {
            "input": str(config.input_path),
            "difficulty": config.difficulty,
            "seed": config.seed,
            "settings": {
                "max_colors": config.max_colors,
                "max_size": config.max_size,
                "min_region_size": config.min_region_size,
                "smooth_radius": config.smooth_radius,
                "starter_palette": config.starter_palette,
                "max_regions": config.max_regions,
            },
            "image_size": {"width": image_size[0], "height": image_size[1]},
            "artifacts": ["outline.svg", "preview.png", "palette.json", "manifest.json"],
            "palette": [
                {"index": entry.index, "hex": entry.hex, "rgb": list(entry.rgb)}
                for entry in palette
            ],
            "regions": [
                {
                    "id": region.id,
                    "palette_index": region.color_index + 1,
                    "area": region.area,
                    "bbox": list(region.bbox),
                    "label": {"x": label_map[region.id].x, "y": label_map[region.id].y},
                }
                for region in regions
            ],
        }
