from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from paintify.config import PaintifyConfig
from paintify.models import LabelPlacement, OutputArtifact, PaintByNumbersDocument, PaletteEntry


class PngPreviewRenderer:
    artifact_name = "preview.png"

    def render(self, config: PaintifyConfig, document: PaintByNumbersDocument) -> OutputArtifact:
        return OutputArtifact(
            self.artifact_name,
            self._render_png(
                document.color_labels,
                document.region_labels,
                document.palette,
                document.labels,
            ),
        )

    def _render_png(
        self,
        color_labels: np.ndarray,
        region_labels: np.ndarray,
        palette: list[PaletteEntry],
        labels: list[LabelPlacement],
    ) -> bytes:
        height, width = color_labels.shape
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        for entry in palette:
            rgb[color_labels == entry.index - 1] = entry.rgb
        edges = np.zeros((height, width), dtype=bool)
        edges[:, 1:] |= region_labels[:, 1:] != region_labels[:, :-1]
        edges[1:, :] |= region_labels[1:, :] != region_labels[:-1, :]
        rgb[edges] = (0, 0, 0)
        scale = 6
        image = Image.fromarray(rgb, mode="RGB").resize(
            (width * scale, height * scale), Image.Resampling.NEAREST
        )
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        for placement in labels:
            text = str(placement.color_index + 1)
            draw.text(
                (placement.x * scale + scale // 2, placement.y * scale + scale // 2),
                text,
                fill=(0, 0, 0),
                anchor="mm",
                font=font,
            )
        output = BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()
