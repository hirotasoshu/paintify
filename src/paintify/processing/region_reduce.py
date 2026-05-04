from __future__ import annotations

import heapq

import cv2
import numpy as np

from paintify.models import Region
from paintify.processing.region_fill import RegionFiller
from paintify.processing.region_table import RegionTopology


class LocalRegionReducer:
    def __init__(self, filler: RegionFiller | None = None) -> None:
        self._filler = filler or RegionFiller()

    def enforce_max_regions(
        self,
        region_labels: np.ndarray,
        color_labels: np.ndarray,
        lab_palette: np.ndarray,
        max_regions: int,
        max_iterations: int = 100_000,
    ) -> tuple[np.ndarray, np.ndarray, list[Region]]:
        if max_regions < 1:
            raise ValueError("max_regions must be positive")

        merged_regions = region_labels.copy()
        merged_colors = color_labels.copy()
        color_distances = self._filler.palette_distances(lab_palette)
        active = {
            region.id: region for region in RegionTopology.table(merged_regions, merged_colors)
        }
        heap = [(region.area, region.id) for region in active.values()]
        heapq.heapify(heap)

        for _ in range(max_iterations):
            if len(active) <= max_regions or len(active) <= 1:
                break
            if not self._reduce_one(merged_regions, merged_colors, color_distances, active, heap):
                break
        return RegionTopology.compact(merged_regions, merged_colors)

    def _reduce_one(
        self,
        region_labels: np.ndarray,
        color_labels: np.ndarray,
        color_distances: np.ndarray,
        active: dict[int, Region],
        heap: list[tuple[int, int]],
    ) -> bool:
        region = self._pop_current_smallest(heap, active)
        if region is None:
            return False

        rebuilt_regions = self._delete_region_locally(
            region_labels,
            color_labels,
            color_distances,
            region,
            active,
        )
        for rebuilt_region in rebuilt_regions:
            heapq.heappush(heap, (rebuilt_region.area, rebuilt_region.id))
        return bool(rebuilt_regions)

    @staticmethod
    def _pop_current_smallest(
        heap: list[tuple[int, int]], active: dict[int, Region]
    ) -> Region | None:
        while heap:
            area, region_id = heapq.heappop(heap)
            region = active.get(region_id)
            if region is not None and region.area == area:
                return region
        return None

    def _delete_region_locally(
        self,
        region_labels: np.ndarray,
        color_labels: np.ndarray,
        color_distances: np.ndarray,
        region: Region,
        active: dict[int, Region],
    ) -> list[Region]:
        neighbors = self._neighbors_for_region(region_labels, region.id, region.bbox)
        if not neighbors:
            return []

        window = RegionTopology.padded_bbox(region.bbox, region_labels.shape)
        local_regions, local_colors = self._local_views(region_labels, color_labels, window)
        remove_mask = local_regions == region.id
        source_mask = np.isin(local_regions, list(neighbors & active.keys()))
        changed = self._filler.fill_removed_mask_from_sources(
            local_regions,
            local_colors,
            remove_mask,
            source_mask,
            color_distances,
        )
        if not changed:
            return []

        affected_ids = {region.id} | (neighbors & active.keys())
        return self._rebuild_affected_regions(region_labels, color_labels, active, affected_ids)

    def _rebuild_affected_regions(
        self,
        region_labels: np.ndarray,
        color_labels: np.ndarray,
        active: dict[int, Region],
        affected_ids: set[int],
    ) -> list[Region]:
        affected_regions = [active[region_id] for region_id in affected_ids if region_id in active]
        if not affected_regions:
            return []

        bbox = self._affected_bbox(affected_regions)
        local_regions, local_colors = self._local_views(region_labels, color_labels, bbox)
        affected_mask = np.isin(local_regions, list(affected_ids))
        if not bool(np.any(affected_mask)):
            return []

        self._remove_active_regions(active, affected_ids)
        local_regions[affected_mask] = 0
        return self._rebuild_components(local_regions, local_colors, affected_mask, active, bbox)

    def _neighbors_for_region(
        self,
        region_labels: np.ndarray,
        region_id: int,
        bbox: tuple[int, int, int, int],
    ) -> set[int]:
        x1, y1, x2, y2 = RegionTopology.padded_bbox(bbox, region_labels.shape)
        local = region_labels[y1:y2, x1:x2]
        neighbors = set(RegionTopology.neighbors(local, region_id))
        neighbors.discard(region_id)
        neighbors.discard(0)
        return neighbors

    @staticmethod
    def _local_views(
        region_labels: np.ndarray,
        color_labels: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        x1, y1, x2, y2 = bbox
        return region_labels[y1:y2, x1:x2], color_labels[y1:y2, x1:x2]

    @staticmethod
    def _affected_bbox(regions: list[Region]) -> tuple[int, int, int, int]:
        bbox = regions[0].bbox
        for region in regions[1:]:
            bbox = RegionTopology.union_bboxes(bbox, region.bbox)
        return bbox

    @staticmethod
    def _remove_active_regions(active: dict[int, Region], affected_ids: set[int]) -> None:
        for region_id in affected_ids:
            active.pop(region_id, None)

    def _rebuild_components(
        self,
        local_regions: np.ndarray,
        local_colors: np.ndarray,
        affected_mask: np.ndarray,
        active: dict[int, Region],
        bbox: tuple[int, int, int, int],
    ) -> list[Region]:
        next_id = int(max(active, default=0)) + 1
        rebuilt_regions: list[Region] = []
        for color_index in sorted(int(value) for value in np.unique(local_colors[affected_mask])):
            rebuilt_regions.extend(
                self._rebuild_color_components(
                    local_regions, local_colors, affected_mask, color_index, active, bbox, next_id
                )
            )
            if rebuilt_regions:
                next_id = max(region.id for region in rebuilt_regions) + 1
        return rebuilt_regions

    @staticmethod
    def _rebuild_color_components(
        local_regions: np.ndarray,
        local_colors: np.ndarray,
        affected_mask: np.ndarray,
        color_index: int,
        active: dict[int, Region],
        bbox: tuple[int, int, int, int],
        next_id: int,
    ) -> list[Region]:
        count, labeled = cv2.connectedComponents(
            (affected_mask & (local_colors == color_index)).astype(np.uint8),
            connectivity=4,
        )
        regions: list[Region] = []
        for component_id in range(1, int(count)):
            region = LocalRegionReducer._component_region(
                labeled, component_id, color_index, bbox, next_id
            )
            local_regions[labeled == component_id] = region.id
            active[region.id] = region
            regions.append(region)
            next_id += 1
        return regions

    @staticmethod
    def _component_region(
        labels: np.ndarray,
        component_id: int,
        color_index: int,
        bbox: tuple[int, int, int, int],
        region_id: int,
    ) -> Region:
        x1, y1, _, _ = bbox
        ys, xs = np.nonzero(labels == component_id)
        return Region(
            id=region_id,
            color_index=color_index,
            area=int(ys.size),
            bbox=(
                int(xs.min()) + x1,
                int(ys.min()) + y1,
                int(xs.max()) + x1 + 1,
                int(ys.max()) + y1 + 1,
            ),
        )
