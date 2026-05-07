import json
from pathlib import Path

import numpy as np

from paintify.processing.quantization import KMeansQuantizer


def test_quantize_respects_bounded_color_count_and_shape() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:4, :, :] = (255, 0, 0)
    image[4:, :, :] = (0, 0, 255)

    labels, palette = KMeansQuantizer().quantize(image, max_colors=2, seed=123, palette_file=None)

    assert labels.shape == (8, 8)
    assert palette.shape[0] <= 2
    assert len({int(value) for value in labels.ravel()}) <= 2


def test_quantize_is_deterministic_for_seed(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    image = rng.integers(0, 255, size=(10, 10, 3), dtype=np.uint8)
    palette_path = tmp_path / "palette.json"
    palette_path.write_text(
        json.dumps({"colors": ["#ffffff", "#000000", "#d62828"]}),
        encoding="utf-8",
    )

    quantizer = KMeansQuantizer()
    first_labels, first_palette = quantizer.quantize(
        image, max_colors=4, seed=5, palette_file=palette_path
    )
    second_labels, second_palette = quantizer.quantize(
        image, max_colors=4, seed=5, palette_file=palette_path
    )

    np.testing.assert_array_equal(first_labels, second_labels)
    np.testing.assert_allclose(first_palette, second_palette)


def test_quantize_uses_all_palette_file_colors_for_negative_one_max_colors(tmp_path: Path) -> None:
    image = np.zeros((9, 9, 3), dtype=np.uint8)
    image[:3, :, :] = (255, 0, 0)
    image[3:6, :, :] = (0, 255, 0)
    image[6:, :, :] = (0, 0, 255)
    palette_path = tmp_path / "palette.json"
    palette_path.write_text(
        json.dumps({"colors": ["#ff0000", "#00ff00", "#0000ff"]}),
        encoding="utf-8",
    )

    labels, palette = KMeansQuantizer().quantize(
        image, max_colors=-1, seed=123, palette_file=palette_path
    )

    assert palette.shape[0] == 3
    assert len({int(value) for value in labels.ravel()}) == 3
