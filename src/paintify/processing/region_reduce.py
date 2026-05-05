from __future__ import annotations

import heapq

import cv2
import numpy as np

from paintify.processing.region_fill import RegionFillContext
from paintify.processing.region_table import Region, RegionMap


class LocalRegionReducer:
    def __init__(self, region_map: RegionMap, fill_context: RegionFillContext) -> None:
        self.region_map = RegionMap(region_map.region_labels.copy(), region_map.color_labels.copy())
        self.fill_context = fill_context
        self.active = {region.id: region for region in self.region_map.regions()}
        self.heap = [(region.area, region.id) for region in self.active.values()]
        heapq.heapify(self.heap)

    def reduce_to(
        self,
        max_regions: int,
        max_iterations: int = 100_000,
    ) -> tuple[np.ndarray, np.ndarray, list[Region]]:
        if max_regions < 1:
            raise ValueError("max_regions must be positive")

        for _ in range(max_iterations):
            if len(self.active) <= max_regions or len(self.active) <= 1:
                break
            if not self._reduce_one():
                break
        if len(self.active) > max_regions:
            raise RuntimeError(
                f"could not reduce regions to {max_regions} within {max_iterations} iterations"
            )
        return self.region_map.compact()

    def _reduce_one(self) -> bool:
        region = self._pop_current_smallest()
        if region is None:
            return False

        rebuilt_regions = self._delete_region_locally(region)
        for rebuilt_region in rebuilt_regions:
            heapq.heappush(self.heap, (rebuilt_region.area, rebuilt_region.id))
        return bool(rebuilt_regions)

    def _pop_current_smallest(self) -> Region | None:
        while self.heap:
            area, region_id = heapq.heappop(self.heap)
            region = self.active.get(region_id)
            if region is not None and region.area == area:
                return region
        return None

    def _delete_region_locally(self, region: Region) -> list[Region]:
        neighbors = self._neighbors_for_region(region.id, region.bbox)
        if not neighbors:
            return []

        window = RegionMap.padded_bbox(region.bbox, self.region_map.region_labels.shape)
        local_map = self.region_map.window(window)
        remove_mask = local_map.region_labels == region.id
        source_mask = np.isin(local_map.region_labels, list(neighbors & self.active.keys()))
        changed = self.fill_context.fill_removed_mask_from_sources(
            local_map,
            remove_mask,
            source_mask,
        )
        if not changed:
            return []

        affected_ids = {region.id} | (neighbors & self.active.keys())
        return self._rebuild_affected_regions(affected_ids)

    def _rebuild_affected_regions(self, affected_ids: set[int]) -> list[Region]:
        affected_regions = [
            self.active[region_id] for region_id in affected_ids if region_id in self.active
        ]
        if not affected_regions:
            return []

        bbox = self._affected_bbox(affected_regions)
        local_map = self.region_map.window(bbox)
        affected_mask = np.isin(local_map.region_labels, list(affected_ids))
        if not bool(np.any(affected_mask)):
            return []

        self._remove_active_regions(affected_ids)
        local_map.region_labels[affected_mask] = 0
        return self._rebuild_components(local_map, affected_mask, bbox)

    def _neighbors_for_region(
        self,
        region_id: int,
        bbox: tuple[int, int, int, int],
    ) -> set[int]:
        window = RegionMap.padded_bbox(bbox, self.region_map.region_labels.shape)
        neighbors = set(self.region_map.window(window).neighbor_counts(region_id))
        neighbors.discard(region_id)
        neighbors.discard(0)
        return neighbors

    @staticmethod
    def _affected_bbox(regions: list[Region]) -> tuple[int, int, int, int]:
        bbox = regions[0].bbox
        for region in regions[1:]:
            bbox = RegionMap.union_bboxes(bbox, region.bbox)
        return bbox

    def _remove_active_regions(self, affected_ids: set[int]) -> None:
        for region_id in affected_ids:
            self.active.pop(region_id, None)

    def _rebuild_components(
        self,
        local_map: RegionMap,
        affected_mask: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> list[Region]:
        next_id = int(max(self.active, default=0)) + 1
        rebuilt_regions: list[Region] = []
        for color_index in sorted(
            int(value) for value in np.unique(local_map.color_labels[affected_mask])
        ):
            rebuilt_regions.extend(
                self._rebuild_color_components(local_map, affected_mask, color_index, bbox, next_id)
            )
            if rebuilt_regions:
                next_id = max(region.id for region in rebuilt_regions) + 1
        return rebuilt_regions

    def _rebuild_color_components(
        self,
        local_map: RegionMap,
        affected_mask: np.ndarray,
        color_index: int,
        bbox: tuple[int, int, int, int],
        next_id: int,
    ) -> list[Region]:
        local_regions = local_map.region_labels
        local_colors = local_map.color_labels
        count, labeled = cv2.connectedComponents(
            (affected_mask & (local_colors == color_index)).astype(np.uint8),
            connectivity=4,
        )
        regions: list[Region] = []
        for component_id in range(1, int(count)):
            region = self._component_region(labeled, component_id, color_index, bbox, next_id)
            local_regions[labeled == component_id] = region.id
            self.active[region.id] = region
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
