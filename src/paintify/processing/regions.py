from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import ndimage  # type: ignore[import-untyped]

from paintify.models import Region


class ConnectedComponentRegionProcessor:
    narrow_strip_cleanup_runs = 3

    def process(
        self,
        color_labels: np.ndarray,
        min_region_size: int,
        max_regions: int | None,
    ) -> tuple[np.ndarray, np.ndarray, list[Region]]:
        color_labels = self.cleanup_narrow_strips(color_labels, runs=self.narrow_strip_cleanup_runs)
        region_labels, _ = self.connected_components(color_labels)
        region_labels, color_labels, regions = self.merge_tiny_regions(
            region_labels,
            color_labels,
            min_region_size,
        )
        if max_regions is not None:
            region_labels, color_labels, regions = self.enforce_max_regions(
                region_labels,
                color_labels,
                max_regions,
            )
        return region_labels, color_labels, regions

    def cleanup_narrow_strips(self, color_labels: np.ndarray, runs: int = 3) -> np.ndarray:
        cleaned = color_labels.copy()
        for _ in range(runs):
            next_labels = cleaned.copy()
            if cleaned.shape[1] >= 3:
                left = cleaned[:, :-2]
                current = cleaned[:, 1:-1]
                right = cleaned[:, 2:]
                horizontal = (left == right) & (current != left)
                next_labels[:, 1:-1][horizontal] = left[horizontal]
            if cleaned.shape[0] >= 3:
                top = cleaned[:-2, :]
                current = cleaned[1:-1, :]
                bottom = cleaned[2:, :]
                vertical = (top == bottom) & (current != top)
                next_labels[1:-1, :][vertical] = top[vertical]
            cleaned = next_labels
            if np.array_equal(cleaned, color_labels):
                break
            color_labels = cleaned
        return cleaned

    def connected_components(self, color_labels: np.ndarray) -> tuple[np.ndarray, list[Region]]:
        next_id = 1
        region_labels = np.zeros_like(color_labels, dtype=np.int32)
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        for color_index in sorted(int(value) for value in np.unique(color_labels)):
            labeled, count = ndimage.label(color_labels == color_index, structure=structure)
            if count == 0:
                continue
            mask = labeled != 0
            region_labels[mask] = labeled[mask] + next_id - 1
            next_id += int(count)
        return region_labels, self._region_table(region_labels, color_labels)

    def merge_tiny_regions(
        self,
        region_labels: np.ndarray,
        color_labels: np.ndarray,
        min_region_size: int,
        max_iterations: int = 10_000,
    ) -> tuple[np.ndarray, np.ndarray, list[Region]]:
        merged_regions = region_labels.copy()
        merged_colors = color_labels.copy()
        for _ in range(max_iterations):
            regions = self._region_table(merged_regions, merged_colors)
            tiny = [region for region in regions if region.area < min_region_size]
            if not tiny or len(regions) <= 1:
                break
            if not self._merge_regions_once(
                merged_regions, merged_colors, tiny, regions, exclude_removed_targets=False
            ):
                break
            merged_regions, merged_colors, regions = self._compact_regions(
                merged_regions, merged_colors
            )
        return self._compact_regions(merged_regions, merged_colors)

    def enforce_max_regions(
        self,
        region_labels: np.ndarray,
        color_labels: np.ndarray,
        max_regions: int,
        max_iterations: int = 100_000,
    ) -> tuple[np.ndarray, np.ndarray, list[Region]]:
        if max_regions < 1:
            raise ValueError("max_regions must be positive")
        merged_regions = region_labels.copy()
        merged_colors = color_labels.copy()
        for _ in range(max_iterations):
            regions = self._region_table(merged_regions, merged_colors)
            if len(regions) <= max_regions or len(regions) <= 1:
                break
            remove_count = max(1, len(regions) - max_regions)
            to_remove = sorted(regions, key=lambda region: (region.area, region.id))[:remove_count]
            if not self._merge_regions_once(
                merged_regions, merged_colors, to_remove, regions, exclude_removed_targets=True
            ):
                break
            merged_regions, merged_colors, regions = self._compact_regions(
                merged_regions, merged_colors
            )
        return self._compact_regions(merged_regions, merged_colors)

    def _merge_regions_once(
        self,
        region_labels: np.ndarray,
        color_labels: np.ndarray,
        regions_to_remove: list[Region],
        regions: list[Region],
        exclude_removed_targets: bool,
    ) -> bool:
        remove_ids = {region.id for region in regions_to_remove}
        return self.fill_removed_regions_from_nearest_kept_pixels(
            region_labels, color_labels, remove_ids
        )

    def fill_removed_regions_from_nearest_kept_pixels(
        self,
        region_labels: np.ndarray,
        color_labels: np.ndarray,
        remove_ids: set[int],
    ) -> bool:
        if not remove_ids:
            return False
        remove_mask = np.isin(region_labels, list(remove_ids))
        if not bool(np.any(remove_mask)):
            return False
        kept_mask = (region_labels != 0) & ~remove_mask
        if not bool(np.any(kept_mask)):
            return False

        ys, xs = np.nonzero(remove_mask)
        old_colors = color_labels[ys, xs]
        best_distances = np.full(ys.shape, np.inf)
        best_color_distances = np.full(ys.shape, np.iinfo(np.int32).max, dtype=np.int32)
        nearest_y = np.zeros(ys.shape, dtype=np.int32)
        nearest_x = np.zeros(xs.shape, dtype=np.int32)

        for color_index in sorted(int(value) for value in np.unique(color_labels[kept_mask])):
            source_mask = kept_mask & (color_labels == color_index)
            distances, indices = ndimage.distance_transform_edt(~source_mask, return_indices=True)
            candidate_distances = distances[ys, xs]
            candidate_color_distances = np.abs(old_colors - color_index).astype(np.int32)
            better = (candidate_distances < best_distances) | (
                (candidate_distances == best_distances)
                & (candidate_color_distances < best_color_distances)
            )
            best_distances[better] = candidate_distances[better]
            best_color_distances[better] = candidate_color_distances[better]
            nearest_y[better] = indices[0, ys[better], xs[better]]
            nearest_x[better] = indices[1, ys[better], xs[better]]

        region_labels[ys, xs] = region_labels[nearest_y, nearest_x]
        color_labels[ys, xs] = color_labels[nearest_y, nearest_x]
        return True

    def _region_table(self, region_labels: np.ndarray, color_labels: np.ndarray) -> list[Region]:
        ys, xs = np.nonzero(region_labels)
        if ys.size == 0:
            return []

        ids = region_labels[ys, xs].astype(np.int32)
        unique_ids, first_indices, counts = np.unique(ids, return_index=True, return_counts=True)
        objects = ndimage.find_objects(region_labels)
        color_indices = color_labels[ys[first_indices], xs[first_indices]]

        return [
            Region(
                id=int(region_id),
                color_index=int(color_index),
                area=int(area),
                bbox=(int(box[1].start), int(box[0].start), int(box[1].stop), int(box[0].stop)),
            )
            for region_id, color_index, area, box in zip(
                unique_ids,
                color_indices,
                counts,
                (objects[int(region_id) - 1] for region_id in unique_ids),
                strict=True,
            )
            if box is not None
        ]

    def _compact_regions(
        self, region_labels: np.ndarray, color_labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, list[Region]]:
        old_ids = np.array(
            sorted(int(v) for v in np.unique(region_labels) if v != 0), dtype=np.int32
        )
        if old_ids.size == 0:
            return np.zeros_like(region_labels, dtype=np.int32), color_labels.copy(), []
        id_map = np.zeros(int(old_ids[-1]) + 1, dtype=np.int32)
        id_map[old_ids] = np.arange(1, old_ids.size + 1, dtype=np.int32)
        compact = id_map[region_labels]
        regions = self._region_table(compact, color_labels)
        color_by_region = np.zeros(int(compact.max()) + 1, dtype=color_labels.dtype)
        for region in regions:
            color_by_region[region.id] = region.color_index
        compact_colors = color_by_region[compact]
        return compact, compact_colors, regions

    def _neighbor_counts(self, region_labels: np.ndarray, region_id: int) -> Counter[int]:
        counts: Counter[int] = Counter()
        mask = region_labels == region_id
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            shifted_mask = np.zeros_like(mask, dtype=bool)
            if dy == 1:
                shifted_mask[:-1, :] = mask[1:, :]
            elif dy == -1:
                shifted_mask[1:, :] = mask[:-1, :]
            elif dx == 1:
                shifted_mask[:, :-1] = mask[:, 1:]
            else:
                shifted_mask[:, 1:] = mask[:, :-1]
            neighbors = region_labels[shifted_mask]
            counts.update(int(value) for value in neighbors if value not in (0, region_id))
        return counts
