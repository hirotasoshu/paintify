from __future__ import annotations

import cv2
import numpy as np

from paintify.models import Region
from paintify.processing.region_fill import RegionFillContext
from paintify.processing.region_reduce import LocalRegionReducer
from paintify.processing.region_table import RegionMap


class ConnectedComponentRegionProcessor:
    narrow_strip_cleanup_runs = 3
    narrow_strip_width = 3

    def process(
        self,
        color_labels: np.ndarray,
        lab_palette: np.ndarray,
        min_region_size: int,
        max_regions: int | None,
    ) -> tuple[np.ndarray, np.ndarray, list[Region]]:
        cleaned_colors = self.cleanup_narrow_strips(
            color_labels, runs=self.narrow_strip_cleanup_runs
        )
        region_labels, _ = self.connected_components(cleaned_colors)
        region_labels, cleaned_colors, regions = self.merge_tiny_regions(
            region_labels,
            cleaned_colors,
            lab_palette,
            min_region_size,
        )
        if max_regions is None:
            return region_labels, cleaned_colors, regions
        return self.enforce_max_regions(region_labels, cleaned_colors, lab_palette, max_regions)

    def cleanup_narrow_strips(self, color_labels: np.ndarray, runs: int = 3) -> np.ndarray:
        cleaned = color_labels.copy()
        for _ in range(runs):
            next_labels = self._cleanup_narrow_strip_run(cleaned)
            if np.array_equal(next_labels, cleaned):
                break
            cleaned = next_labels
        return cleaned

    def connected_components(self, color_labels: np.ndarray) -> tuple[np.ndarray, list[Region]]:
        next_id = 1
        region_labels = np.zeros_like(color_labels, dtype=np.int32)
        for color_index in sorted(int(value) for value in np.unique(color_labels)):
            count, labeled = cv2.connectedComponents(
                (color_labels == color_index).astype(np.uint8), connectivity=4
            )
            if count <= 1:
                continue
            mask = labeled != 0
            region_labels[mask] = labeled[mask] + next_id - 1
            next_id += int(count) - 1
        return region_labels, RegionMap(region_labels, color_labels).regions()

    def merge_tiny_regions(
        self,
        region_labels: np.ndarray,
        color_labels: np.ndarray,
        lab_palette: np.ndarray,
        min_region_size: int,
        max_iterations: int = 10_000,
    ) -> tuple[np.ndarray, np.ndarray, list[Region]]:
        merged_regions = region_labels.copy()
        merged_colors = color_labels.copy()
        fill_context = RegionFillContext(lab_palette)
        for _ in range(max_iterations):
            region_map = RegionMap(merged_regions, merged_colors)
            regions = region_map.regions()
            tiny = [region for region in regions if region.area < min_region_size]
            if not tiny or len(regions) <= 1:
                break
            changed = fill_context.fill_removed_regions_from_nearest_kept_pixels(
                region_map, {region.id for region in tiny}
            )
            if not changed:
                break
            merged_regions, merged_colors, regions = region_map.compact()
        return RegionMap(merged_regions, merged_colors).compact()

    def enforce_max_regions(
        self,
        region_labels: np.ndarray,
        color_labels: np.ndarray,
        lab_palette: np.ndarray,
        max_regions: int,
        max_iterations: int = 100_000,
    ) -> tuple[np.ndarray, np.ndarray, list[Region]]:
        return LocalRegionReducer(
            RegionMap(region_labels, color_labels), RegionFillContext(lab_palette)
        ).reduce_to(
            max_regions,
            max_iterations,
        )

    @staticmethod
    def _cleanup_narrow_strip_run(color_labels: np.ndarray) -> np.ndarray:
        next_labels = color_labels.copy()
        ConnectedComponentRegionProcessor._cleanup_horizontal_strips(color_labels, next_labels)
        ConnectedComponentRegionProcessor._cleanup_vertical_strips(color_labels, next_labels)
        return next_labels

    @staticmethod
    def _cleanup_horizontal_strips(color_labels: np.ndarray, next_labels: np.ndarray) -> None:
        if color_labels.shape[1] < ConnectedComponentRegionProcessor.narrow_strip_width:
            return
        left = color_labels[:, :-2]
        current = color_labels[:, 1:-1]
        right = color_labels[:, 2:]
        horizontal = (left == right) & (current != left)
        next_labels[:, 1:-1][horizontal] = left[horizontal]

    @staticmethod
    def _cleanup_vertical_strips(color_labels: np.ndarray, next_labels: np.ndarray) -> None:
        if color_labels.shape[0] < ConnectedComponentRegionProcessor.narrow_strip_width:
            return
        top = color_labels[:-2, :]
        current = color_labels[1:-1, :]
        bottom = color_labels[2:, :]
        vertical = (top == bottom) & (current != top)
        next_labels[1:-1, :][vertical] = top[vertical]
