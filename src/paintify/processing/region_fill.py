from __future__ import annotations

import cv2
import numpy as np

from paintify.processing.region_table import RegionMap


class RegionFillContext:
    def __init__(self, lab_palette: np.ndarray) -> None:
        self.color_distances = self._palette_distances(lab_palette)

    @staticmethod
    def _palette_distances(lab_palette: np.ndarray) -> np.ndarray:
        palette = lab_palette.astype(np.float32, copy=False)
        diff = palette[:, None, :] - palette[None, :, :]
        return np.linalg.norm(diff, axis=2)

    def fill_removed_regions_from_nearest_kept_pixels(
        self,
        region_map: RegionMap,
        remove_ids: set[int],
    ) -> bool:
        if not remove_ids:
            return False

        remove_mask = np.isin(region_map.region_labels, list(remove_ids))
        kept_mask = (region_map.region_labels != 0) & ~remove_mask
        return self.fill_removed_mask_from_sources(
            region_map,
            remove_mask,
            kept_mask,
        )

    def fill_removed_mask_from_sources(
        self,
        region_map: RegionMap,
        remove_mask: np.ndarray,
        source_mask: np.ndarray,
    ) -> bool:
        ys, xs = np.nonzero(remove_mask)
        if ys.size == 0 or not bool(np.any(source_mask)):
            return False

        self._fill_removed_points(
            region_map,
            source_mask,
            ys,
            xs,
        )
        return True

    def _fill_removed_points(
        self,
        region_map: RegionMap,
        source_mask: np.ndarray,
        ys: np.ndarray,
        xs: np.ndarray,
    ) -> None:
        region_labels = region_map.region_labels
        color_labels = region_map.color_labels
        old_colors = color_labels[ys, xs]
        best_distances = np.full(ys.shape, np.inf)
        best_color_distances = np.full(ys.shape, np.inf, dtype=np.float32)

        for color_index in sorted(int(value) for value in np.unique(color_labels[source_mask])):
            candidate_source = source_mask & (color_labels == color_index)
            distances, nearest_regions, nearest_colors = self._nearest_source_values(
                candidate_source, region_labels, color_labels
            )
            candidate_distances = distances[ys, xs]
            candidate_color_distances = self.color_distances[old_colors, color_index]
            better = (candidate_distances < best_distances) | (
                (candidate_distances == best_distances)
                & (candidate_color_distances < best_color_distances)
            )
            best_distances[better] = candidate_distances[better]
            best_color_distances[better] = candidate_color_distances[better]
            region_labels[ys[better], xs[better]] = nearest_regions[ys[better], xs[better]]
            color_labels[ys[better], xs[better]] = nearest_colors[ys[better], xs[better]]

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
        unique_label_ids, first_indices = np.unique(label_ids[order], return_index=True)
        source_indices = order[first_indices]
        nearest_y = source_y[source_indices]
        nearest_x = source_x[source_indices]

        nearest_regions_by_label = np.zeros(int(labels.max()) + 1, dtype=region_labels.dtype)
        nearest_colors_by_label = np.zeros(int(labels.max()) + 1, dtype=color_labels.dtype)
        nearest_regions_by_label[unique_label_ids] = region_labels[nearest_y, nearest_x]
        nearest_colors_by_label[unique_label_ids] = color_labels[nearest_y, nearest_x]
        return distances, nearest_regions_by_label[labels], nearest_colors_by_label[labels]
