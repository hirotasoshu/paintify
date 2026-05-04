from __future__ import annotations

import time

import numpy as np

from paintify.processing.regions import ConnectedComponentRegionProcessor


def test_connected_components_splits_same_color_islands() -> None:
    labels = np.array([[0, 1, 0]], dtype=np.int32)

    region_labels, regions = ConnectedComponentRegionProcessor().connected_components(labels)

    assert region_labels.max() == 3
    assert [region.area for region in regions] == [1, 1, 1]


def test_connected_components_handles_dense_region_maps_quickly() -> None:
    labels = np.indices((240, 240)).sum(axis=0).astype(np.int32) % 2

    start = time.perf_counter()
    region_labels, regions = ConnectedComponentRegionProcessor().connected_components(labels)
    elapsed = time.perf_counter() - start

    assert int(region_labels.max()) == labels.size
    assert len(regions) == labels.size
    assert elapsed < 0.2


def test_region_table_handles_many_large_regions_quickly() -> None:
    y_indices, x_indices = np.indices((1000, 1000))
    region_labels = (y_indices // 25 * 40 + x_indices // 25 + 1).astype(np.int32)
    color_labels = (region_labels % 30).astype(np.int32)

    start = time.perf_counter()
    regions = ConnectedComponentRegionProcessor()._region_table(region_labels, color_labels)
    elapsed = time.perf_counter() - start

    assert len(regions) == 1600
    assert sum(region.area for region in regions) == region_labels.size
    assert elapsed < 0.12


def test_tiny_region_merge_removes_small_neighbor_region() -> None:
    color_labels = np.zeros((5, 5), dtype=np.int32)
    color_labels[2, 2] = 1
    processor = ConnectedComponentRegionProcessor()
    region_labels, _ = processor.connected_components(color_labels)

    merged_labels, merged_colors, regions = processor.merge_tiny_regions(
        region_labels, color_labels, min_region_size=2
    )

    assert len(regions) == 1
    assert regions[0].area == 25
    assert int(merged_labels.max()) == 1
    assert set(int(value) for value in merged_colors.ravel()) == {0}


def test_process_removes_one_pixel_strips_before_building_regions() -> None:
    color_labels = np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=np.int32,
    )

    region_labels, merged_colors, regions = ConnectedComponentRegionProcessor().process(
        color_labels,
        min_region_size=1,
        max_regions=None,
    )

    assert len(regions) == 1
    assert int(region_labels.max()) == 1
    assert set(int(value) for value in merged_colors.ravel()) == {0}


def test_tiny_region_merge_keeps_total_area_invariant() -> None:
    color_labels = np.array(
        [
            [0, 0, 1, 1],
            [0, 2, 2, 1],
            [0, 2, 1, 1],
        ],
        dtype=np.int32,
    )
    processor = ConnectedComponentRegionProcessor()
    region_labels, _ = processor.connected_components(color_labels)

    _, _, regions = processor.merge_tiny_regions(region_labels, color_labels, min_region_size=3)

    assert sum(region.area for region in regions) == color_labels.size
    assert all(region.area >= 3 for region in regions)


def test_max_regions_merges_smallest_regions_and_compacts_ids() -> None:
    color_labels = np.array(
        [
            [0, 0, 1, 1],
            [0, 2, 2, 1],
            [3, 3, 2, 1],
        ],
        dtype=np.int32,
    )
    processor = ConnectedComponentRegionProcessor()
    region_labels, _ = processor.connected_components(color_labels)

    merged_labels, merged_colors, regions = processor.enforce_max_regions(
        region_labels, color_labels, max_regions=2
    )

    assert len(regions) == 2
    assert sum(region.area for region in regions) == color_labels.size
    assert sorted(int(value) for value in np.unique(merged_labels)) == [1, 2]
    assert {region.id for region in regions} == {1, 2}
    assert set(int(value) for value in np.unique(merged_colors)) <= {
        region.color_index for region in regions
    }


def test_max_regions_uses_nearest_neighbor_color_tie_break() -> None:
    region_labels = np.array(
        [
            [1, 1, 4, 2],
            [1, 3, 3, 2],
            [1, 2, 2, 2],
        ],
        dtype=np.int32,
    )
    color_labels = region_labels.copy()

    merged_labels, merged_colors, regions = ConnectedComponentRegionProcessor().enforce_max_regions(
        region_labels,
        color_labels,
        max_regions=3,
    )

    assert len(regions) == 3
    assert int(merged_colors[0, 2]) == 3
    assert int(merged_labels[0, 2]) == int(merged_labels[1, 2])


def test_max_regions_prefers_color_distance_over_region_id_tie() -> None:
    region_labels = np.array([[2, 2, 4, 1, 1]], dtype=np.int32)
    color_labels = region_labels.copy()

    merged_labels, merged_colors, regions = ConnectedComponentRegionProcessor().enforce_max_regions(
        region_labels,
        color_labels,
        max_regions=2,
    )

    assert len(regions) == 2
    assert int(merged_colors[0, 2]) == 2
    assert int(merged_labels[0, 2]) == int(merged_labels[0, 1])


def test_tiny_region_pixels_are_reassigned_to_nearest_neighbor_regions() -> None:
    region_labels = np.array(
        [
            [1, 1, 1, 3, 3, 2, 2, 2],
            [1, 1, 1, 3, 3, 2, 2, 2],
            [1, 1, 1, 3, 3, 2, 2, 2],
        ],
        dtype=np.int32,
    )
    color_labels = np.array(
        [
            [0, 0, 0, 1, 1, 2, 2, 2],
            [0, 0, 0, 1, 1, 2, 2, 2],
            [0, 0, 0, 1, 1, 2, 2, 2],
        ],
        dtype=np.int32,
    )

    _, merged_colors, regions = ConnectedComponentRegionProcessor().merge_tiny_regions(
        region_labels,
        color_labels,
        min_region_size=7,
    )

    assert len(regions) == 2
    assert set(int(value) for value in merged_colors[:, 3]) == {0}
    assert set(int(value) for value in merged_colors[:, 4]) == {2}


def test_nearest_neighbor_merge_breaks_distance_ties_by_color_distance() -> None:
    region_labels = np.array([[1, 1, 3, 2, 2], [1, 1, 3, 2, 2]], dtype=np.int32)
    color_labels = np.array([[0, 0, 1, 2, 2], [0, 0, 1, 2, 2]], dtype=np.int32)

    _, merged_colors, regions = ConnectedComponentRegionProcessor().merge_tiny_regions(
        region_labels,
        color_labels,
        min_region_size=3,
    )

    assert len(regions) == 2
    assert int(merged_colors[0, 2]) == 0


def test_removed_regions_can_be_filled_from_nearest_kept_pixels_in_one_batch() -> None:
    region_labels = np.array([[1, 1, 3, 4, 5, 2, 2]], dtype=np.int32)
    color_labels = np.array([[0, 0, 3, 4, 5, 2, 2]], dtype=np.int32)

    changed = ConnectedComponentRegionProcessor().fill_removed_regions_from_nearest_kept_pixels(
        region_labels,
        color_labels,
        remove_ids={3, 4, 5},
    )

    assert changed
    assert np.array_equal(color_labels, np.array([[0, 0, 0, 2, 2, 2, 2]], dtype=np.int32))
