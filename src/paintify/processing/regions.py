from __future__ import annotations

import cv2
import numpy as np

from paintify.models import Region
from paintify.processing.region_fill import RegionFiller
from paintify.processing.region_reduce import LocalRegionReducer
from paintify.processing.region_table import RegionTopology


class ConnectedComponentRegionProcessor:
    narrow_strip_cleanup_runs = 3

    def __init__(self) -> None:
        self._filler = RegionFiller()
        self._max_region_reducer = LocalRegionReducer(self._filler)

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
        return region_labels, self._region_table(region_labels, color_labels)

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
        color_distances = self._palette_distances(lab_palette)
        for _ in range(max_iterations):
            regions = self._region_table(merged_regions, merged_colors)
            tiny = [region for region in regions if region.area < min_region_size]
            if not tiny or len(regions) <= 1:
                break
            changed = self._merge_regions_once(merged_regions, merged_colors, color_distances, tiny)
            if not changed:
                break
            merged_regions, merged_colors, regions = self._compact_regions(
                merged_regions, merged_colors
            )
        return self._compact_regions(merged_regions, merged_colors)

    def enforce_max_regions(
        self,
        region_labels: np.ndarray,
        color_labels: np.ndarray,
        lab_palette: np.ndarray,
        max_regions: int,
        max_iterations: int = 100_000,
    ) -> tuple[np.ndarray, np.ndarray, list[Region]]:
        return self._max_region_reducer.enforce_max_regions(
            region_labels,
            color_labels,
            lab_palette,
            max_regions,
            max_iterations,
        )

    def fill_removed_regions_from_nearest_kept_pixels(
        self,
        region_labels: np.ndarray,
        color_labels: np.ndarray,
        color_distances: np.ndarray,
        remove_ids: set[int],
    ) -> bool:
        return self._filler.fill_removed_regions_from_nearest_kept_pixels(
            region_labels,
            color_labels,
            color_distances,
            remove_ids,
        )

    @staticmethod
    def _cleanup_narrow_strip_run(color_labels: np.ndarray) -> np.ndarray:
        next_labels = color_labels.copy()
        ConnectedComponentRegionProcessor._cleanup_horizontal_strips(color_labels, next_labels)
        ConnectedComponentRegionProcessor._cleanup_vertical_strips(color_labels, next_labels)
        return next_labels

    @staticmethod
    def _cleanup_horizontal_strips(color_labels: np.ndarray, next_labels: np.ndarray) -> None:
        if color_labels.shape[1] < 3:
            return
        left = color_labels[:, :-2]
        current = color_labels[:, 1:-1]
        right = color_labels[:, 2:]
        horizontal = (left == right) & (current != left)
        next_labels[:, 1:-1][horizontal] = left[horizontal]

    @staticmethod
    def _cleanup_vertical_strips(color_labels: np.ndarray, next_labels: np.ndarray) -> None:
        if color_labels.shape[0] < 3:
            return
        top = color_labels[:-2, :]
        current = color_labels[1:-1, :]
        bottom = color_labels[2:, :]
        vertical = (top == bottom) & (current != top)
        next_labels[1:-1, :][vertical] = top[vertical]

    def _merge_regions_once(
        self,
        region_labels: np.ndarray,
        color_labels: np.ndarray,
        color_distances: np.ndarray,
        regions_to_remove: list[Region],
    ) -> bool:
        remove_ids = {region.id for region in regions_to_remove}
        return self.fill_removed_regions_from_nearest_kept_pixels(
            region_labels, color_labels, color_distances, remove_ids
        )

    @staticmethod
    def _palette_distances(lab_palette: np.ndarray) -> np.ndarray:
        return RegionFiller.palette_distances(lab_palette)

    @staticmethod
    def _region_table(region_labels: np.ndarray, color_labels: np.ndarray) -> list[Region]:
        return RegionTopology.table(region_labels, color_labels)

    @staticmethod
    def _compact_regions(
        region_labels: np.ndarray, color_labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, list[Region]]:
        return RegionTopology.compact(region_labels, color_labels)

    @staticmethod
    def _neighbor_counts(region_labels: np.ndarray, region_id: int):
        return RegionTopology.neighbors(region_labels, region_id)
