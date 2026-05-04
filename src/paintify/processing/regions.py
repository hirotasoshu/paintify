from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

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

        for color_index in sorted(int(value) for value in np.unique(color_labels[kept_mask])):
            source_mask = kept_mask & (color_labels == color_index)
            distances, nearest_region_labels, nearest_color_labels = self._nearest_source_values(
                source_mask, region_labels, color_labels
            )
            candidate_distances = distances[ys, xs]
            candidate_color_distances = np.abs(old_colors - color_index).astype(np.int32)
            better = (candidate_distances < best_distances) | (
                (candidate_distances == best_distances)
                & (candidate_color_distances < best_color_distances)
            )
            best_distances[better] = candidate_distances[better]
            best_color_distances[better] = candidate_color_distances[better]
            region_labels[ys[better], xs[better]] = nearest_region_labels[ys[better], xs[better]]
            color_labels[ys[better], xs[better]] = nearest_color_labels[ys[better], xs[better]]

        return True

    @staticmethod
    def _nearest_source_values(
        source_mask: np.ndarray,
        region_labels: np.ndarray,
        color_labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        distances, labels = cv2.distanceTransformWithLabels(
            (~source_mask).astype(np.uint8),
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE,
            labelType=cv2.DIST_LABEL_PIXEL,
        )
        source_y, source_x = np.nonzero(source_mask)
        label_ids = labels[source_y, source_x]
        order = np.argsort(label_ids, kind="stable")
        sorted_label_ids = label_ids[order]
        unique_label_ids, first_indices = np.unique(sorted_label_ids, return_index=True)
        source_indices = order[first_indices]
        max_label = int(labels.max())
        nearest_regions_by_label = np.zeros(max_label + 1, dtype=region_labels.dtype)
        nearest_colors_by_label = np.zeros(max_label + 1, dtype=color_labels.dtype)
        nearest_y = source_y[source_indices]
        nearest_x = source_x[source_indices]
        nearest_regions_by_label[unique_label_ids] = region_labels[nearest_y, nearest_x]
        nearest_colors_by_label[unique_label_ids] = color_labels[nearest_y, nearest_x]
        return distances, nearest_regions_by_label[labels], nearest_colors_by_label[labels]

    def _region_table(self, region_labels: np.ndarray, color_labels: np.ndarray) -> list[Region]:
        ys, xs = np.nonzero(region_labels)
        if ys.size == 0:
            return []

        ids = region_labels[ys, xs].astype(np.int32)
        order = np.argsort(ids, kind="stable")
        sorted_ids = ids[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_ids)) + 1]
        unique_ids = sorted_ids[starts]
        sorted_ys = ys[order]
        sorted_xs = xs[order]
        counts = np.diff(np.r_[starts, sorted_ids.size])
        min_x = np.minimum.reduceat(sorted_xs, starts)
        min_y = np.minimum.reduceat(sorted_ys, starts)
        max_x = np.maximum.reduceat(sorted_xs, starts)
        max_y = np.maximum.reduceat(sorted_ys, starts)
        color_indices = color_labels[sorted_ys[starts], sorted_xs[starts]]

        return [
            Region(
                id=int(region_id),
                color_index=int(color_index),
                area=int(area),
                bbox=(int(x1), int(y1), int(x2) + 1, int(y2) + 1),
            )
            for region_id, color_index, area, x1, y1, x2, y2 in zip(
                unique_ids,
                color_indices,
                counts,
                min_x,
                min_y,
                max_x,
                max_y,
                strict=True,
            )
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
