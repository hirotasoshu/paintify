from __future__ import annotations

import cv2
import numpy as np

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
        image = cv2.resize(
            rgb,
            (width * scale, height * scale),
            interpolation=cv2.INTER_NEAREST,
        )
        for placement in labels:
            text = str(placement.color_index + 1)
            cv2.putText(
                image,
                text,
                (placement.x * scale + 1, placement.y * scale + scale - 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.25,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        success, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not success:
            raise ValueError("failed to encode PNG preview")
        return encoded.tobytes()
