from __future__ import annotations

from xml.sax.saxutils import escape

import numpy as np

from paintify.config import PaintifyConfig
from paintify.models import LabelPlacement, OutputArtifact, PaintByNumbersDocument


class SvgOutlineRenderer:
    artifact_name = "outline.svg"

    def render(self, config: PaintifyConfig, document: PaintByNumbersDocument) -> OutputArtifact:
        return OutputArtifact(
            self.artifact_name, self._render_svg(document.region_labels, document.labels)
        )

    def _render_svg(self, region_labels: np.ndarray, labels: list[LabelPlacement]) -> str:
        height, width = region_labels.shape
        scale = 6
        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width * scale}" height="{height * scale}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<g stroke="black" stroke-width="0.08" fill="none" stroke-linecap="square">',
        ]
        for x1, y1, x2, y2 in self._boundary_segments(region_labels):
            lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"/>')
        lines.append("</g>")
        lines.append(
            '<g font-family="Arial, sans-serif" font-size="1.8" text-anchor="middle" dominant-baseline="central" fill="black">'
        )
        for placement in labels:
            lines.append(
                f'<text x="{placement.x + 0.5}" y="{placement.y + 0.5}">{escape(str(placement.color_index + 1))}</text>'
            )
        lines.append("</g></svg>")
        return "\n".join(lines)

    @staticmethod
    def _boundary_segments(region_labels: np.ndarray) -> list[tuple[int, int, int, int]]:
        height, width = region_labels.shape
        segments: list[tuple[int, int, int, int]] = []
        for y in range(height):
            for x in range(width):
                current = int(region_labels[y, x])
                if x == 0 or int(region_labels[y, x - 1]) != current:
                    segments.append((x, y, x, y + 1))
                if y == 0 or int(region_labels[y - 1, x]) != current:
                    segments.append((x, y, x + 1, y))
                if x == width - 1:
                    segments.append((x + 1, y, x + 1, y + 1))
                if y == height - 1:
                    segments.append((x, y + 1, x + 1, y + 1))
        return segments
