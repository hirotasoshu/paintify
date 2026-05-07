import time

import numpy as np

from paintify.processing.labels import DistanceTransformLabelPlacer
from paintify.processing.regions import ConnectedComponentRegionProcessor


def test_label_placement_stays_inside_region() -> None:
    color_labels = np.zeros((7, 7), dtype=np.int32)
    color_labels[0, :] = 1
    region_labels, regions = ConnectedComponentRegionProcessor().connected_components(color_labels)

    labels = DistanceTransformLabelPlacer().place(region_labels, regions)

    for label in labels:
        assert region_labels[label.y, label.x] == label.region_id


def test_label_placement_prefers_region_interior() -> None:
    color_labels = np.zeros((5, 5), dtype=np.int32)
    region_labels, regions = ConnectedComponentRegionProcessor().connected_components(color_labels)

    label = DistanceTransformLabelPlacer().place(region_labels, regions)[0]

    assert (label.x, label.y) == (2, 2)


def test_label_placement_handles_many_regions_quickly() -> None:
    block_size = 20
    grid_size = 20
    y_indices, x_indices = np.indices((block_size * grid_size, block_size * grid_size))
    color_labels = (y_indices // block_size * grid_size + x_indices // block_size).astype(np.int32)
    region_labels, regions = ConnectedComponentRegionProcessor().connected_components(color_labels)

    start = time.perf_counter()
    labels = DistanceTransformLabelPlacer().place(region_labels, regions)
    elapsed = time.perf_counter() - start

    assert len(labels) == grid_size * grid_size
    assert elapsed < 0.08
