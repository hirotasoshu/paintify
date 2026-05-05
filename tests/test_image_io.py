from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from paintify.processing.image import ImageInputError, OpenCvImageLoader


def test_load_image_resizes_and_keeps_rgb_channel_order(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    image = np.zeros((4, 8, 3), dtype=np.uint8)
    image[:, :] = (255, 0, 0)
    assert cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    loaded = OpenCvImageLoader().load(image_path, max_size=4, smooth_radius=0)

    assert loaded.shape[:2] == (2, 4)
    assert tuple(int(value) for value in loaded[0, 0]) == (255, 0, 0)


def test_load_image_clamps_extreme_aspect_resize_to_one_pixel(tmp_path: Path) -> None:
    image_path = tmp_path / "wide.png"
    image = np.zeros((1, 1000, 3), dtype=np.uint8)
    assert cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    loaded = OpenCvImageLoader().load(image_path, max_size=8, smooth_radius=0)

    assert loaded.shape[:2] == (1, 8)


def test_load_image_reports_unreadable_image_as_input_error(tmp_path: Path) -> None:
    image_path = tmp_path / "not-image.txt"
    image_path.write_text("definitely not an image", encoding="utf-8")

    with pytest.raises(ImageInputError, match="is not a readable image file"):
        OpenCvImageLoader().load(image_path, max_size=10, smooth_radius=0)
