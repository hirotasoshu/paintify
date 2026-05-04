from __future__ import annotations

from xml.sax.saxutils import escape

import numpy as np

from paintify.config import PaintifyConfig
from paintify.models import LabelPlacement, OutputArtifact, PaintByNumbersDocument


class SvgOutlineRenderer:
    artifact_name = "outline.svg"
    scale = 6

    def render(self, config: PaintifyConfig, document: PaintByNumbersDocument) -> OutputArtifact:
        return OutputArtifact(
            self.artifact_name, self._render_svg(document.region_labels, document.labels)
        )

    def _render_svg(self, region_labels: np.ndarray, labels: list[LabelPlacement]) -> str:
        lines = [
            self._svg_header(region_labels),
            '<rect width="100%" height="100%" fill="white"/>',
            '<g stroke="black" stroke-width="0.08" fill="none" stroke-linecap="square">',
        ]
        lines.extend(self._line_elements(region_labels))
        lines.append("</g>")
        lines.extend(self._label_elements(labels))
        return "\n".join(lines)

    def _svg_header(self, region_labels: np.ndarray) -> str:
        height, width = region_labels.shape
        scaled_width = width * self.scale
        scaled_height = height * self.scale
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{scaled_width}" '
            f'height="{scaled_height}" viewBox="0 0 {width} {height}">'
        )

    def _line_elements(self, region_labels: np.ndarray) -> list[str]:
        return [
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"/>'
            for x1, y1, x2, y2 in self._boundary_segments(region_labels)
        ]

    def _label_elements(self, labels: list[LabelPlacement]) -> list[str]:
        lines = [
            '<g font-family="Arial, sans-serif" font-size="1.8" '
            'text-anchor="middle" dominant-baseline="central" fill="black">'
        ]
        lines.extend(self._text_element(placement) for placement in labels)
        lines.append("</g></svg>")
        return lines

    @staticmethod
    def _text_element(placement: LabelPlacement) -> str:
        text = escape(str(placement.color_index + 1))
        return f'<text x="{placement.x + 0.5}" y="{placement.y + 0.5}">{text}</text>'

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
