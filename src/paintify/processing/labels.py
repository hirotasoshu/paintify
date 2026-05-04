from __future__ import annotations

import numpy as np
from scipy import ndimage  # type: ignore[import-untyped]

from paintify.models import LabelPlacement, Region


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
        return ndimage.distance_transform_edt(padded)[1:-1, 1:-1]
