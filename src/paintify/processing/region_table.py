from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Region:
    id: int
    color_index: int
    area: int
    bbox: tuple[int, int, int, int]


class RegionMap:
    def __init__(self, region_labels: np.ndarray, color_labels: np.ndarray) -> None:
        self.region_labels = region_labels
        self.color_labels = color_labels

    def regions(self) -> list[Region]:
        ys, xs = np.nonzero(self.region_labels)
        if ys.size == 0:
            return []

        ids = self.region_labels[ys, xs].astype(np.int32)
        order = np.argsort(ids, kind="stable")
        sorted_ids = ids[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_ids)) + 1]
        sorted_ys = ys[order]
        sorted_xs = xs[order]
        counts = np.diff(np.r_[starts, sorted_ids.size])

        return self._regions_from_groups(
            sorted_ids,
            sorted_ys,
            sorted_xs,
            self.color_labels,
            starts,
            counts,
        )

    def compact(self) -> tuple[np.ndarray, np.ndarray, list[Region]]:
        old_ids = np.array(
            sorted(int(value) for value in np.unique(self.region_labels) if value != 0),
            dtype=np.int32,
        )
        if old_ids.size == 0:
            return np.zeros_like(self.region_labels, dtype=np.int32), self.color_labels.copy(), []

        id_map = np.zeros(int(old_ids[-1]) + 1, dtype=np.int32)
        id_map[old_ids] = np.arange(1, old_ids.size + 1, dtype=np.int32)
        compact_labels = id_map[self.region_labels]
        compact_map = RegionMap(compact_labels, self.color_labels)
        regions = compact_map.regions()
        compact_colors = self._colors_for_regions(compact_labels, self.color_labels, regions)
        return compact_labels, compact_colors, regions

    def neighbor_counts(self, region_id: int) -> Counter[int]:
        counts: Counter[int] = Counter()
        mask = self.region_labels == region_id
        for shifted_mask in self._shifted_masks(mask):
            neighbors = self.region_labels[shifted_mask]
            counts.update(int(value) for value in neighbors if value not in (0, region_id))
        return counts

    def window(self, bbox: tuple[int, int, int, int]) -> "RegionMap":
        x1, y1, x2, y2 = bbox
        return RegionMap(self.region_labels[y1:y2, x1:x2], self.color_labels[y1:y2, x1:x2])

    @staticmethod
    def padded_bbox(
        bbox: tuple[int, int, int, int], shape: tuple[int, int]
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        height, width = shape
        return max(0, x1 - 1), max(0, y1 - 1), min(width, x2 + 1), min(height, y2 + 1)

    @staticmethod
    def union_bboxes(
        first: tuple[int, int, int, int], second: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        return (
            min(first[0], second[0]),
            min(first[1], second[1]),
            max(first[2], second[2]),
            max(first[3], second[3]),
        )

    @staticmethod
    def _regions_from_groups(
        sorted_ids: np.ndarray,
        sorted_ys: np.ndarray,
        sorted_xs: np.ndarray,
        color_labels: np.ndarray,
        starts: np.ndarray,
        counts: np.ndarray,
    ) -> list[Region]:
        min_x = np.minimum.reduceat(sorted_xs, starts)
        min_y = np.minimum.reduceat(sorted_ys, starts)
        max_x = np.maximum.reduceat(sorted_xs, starts)
        max_y = np.maximum.reduceat(sorted_ys, starts)
        color_indices = color_labels[sorted_ys[starts], sorted_xs[starts]]
        unique_ids = sorted_ids[starts]

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

    @staticmethod
    def _colors_for_regions(
        compact_labels: np.ndarray, color_labels: np.ndarray, regions: list[Region]
    ) -> np.ndarray:
        color_by_region = np.zeros(int(compact_labels.max()) + 1, dtype=color_labels.dtype)
        for region in regions:
            color_by_region[region.id] = region.color_index
        return color_by_region[compact_labels]

    @staticmethod
    def _shifted_masks(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        down = np.zeros_like(mask, dtype=bool)
        up = np.zeros_like(mask, dtype=bool)
        right = np.zeros_like(mask, dtype=bool)
        left = np.zeros_like(mask, dtype=bool)
        down[:-1, :] = mask[1:, :]
        up[1:, :] = mask[:-1, :]
        right[:, :-1] = mask[:, 1:]
        left[:, 1:] = mask[:, :-1]
        return down, up, right, left
