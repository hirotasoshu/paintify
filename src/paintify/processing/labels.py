from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from paintify.processing.region_table import Region


@dataclass(frozen=True)
class LabelPlacement:
    region_id: int
    color_index: int
    x: int
    y: int


class DistanceTransformLabelPlacer:
    def place(self, region_labels: np.ndarray, regions: list[Region]) -> list[LabelPlacement]:
        placements: list[LabelPlacement] = []
        image_center = np.array(
            [(region_labels.shape[0] - 1) / 2.0, (region_labels.shape[1] - 1) / 2.0]
        )
        for region in regions:
            min_x, min_y, max_x, max_y = region.bbox
            mask = region_labels[min_y:max_y, min_x:max_x] == region.id
            distances = self._distance_to_outside(mask)
            max_distance = float(np.max(distances))
            candidates = np.argwhere((distances == max_distance) & mask)
            global_candidates = candidates + np.array([min_y, min_x], dtype=np.int32)
            center_distances = np.linalg.norm(
                global_candidates.astype(float) - image_center, axis=1
            )
            y, x = global_candidates[int(np.argmin(center_distances))]
            placements.append(
                LabelPlacement(
                    region_id=region.id, color_index=region.color_index, x=int(x), y=int(y)
                )
            )
        return placements

    @staticmethod
    def _distance_to_outside(mask: np.ndarray) -> np.ndarray:
        padded = np.pad(mask, 1, constant_values=False)
        distances = cv2.distanceTransform(
            padded.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        return distances[1:-1, 1:-1]
