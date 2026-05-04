from __future__ import annotations

from pathlib import Path

from PIL import Image

import pytest

from paintify.processing.image import ImageInputError, PillowImageLoader


def test_load_image_applies_exif_orientation(tmp_path: Path) -> None:
    image_path = tmp_path / "oriented.jpg"
    image = Image.new("RGB", (2, 4), color=(255, 0, 0))
    exif = Image.Exif()
    exif[274] = 6
    image.save(image_path, exif=exif)

    loaded = PillowImageLoader().load(image_path, max_size=10, smooth_radius=0)

    assert loaded.shape[:2] == (2, 4)


def test_load_image_reports_unreadable_image_as_input_error(tmp_path: Path) -> None:
    image_path = tmp_path / "not-image.txt"
    image_path.write_text("definitely not an image", encoding="utf-8")

    with pytest.raises(ImageInputError, match="is not a readable image file"):
        PillowImageLoader().load(image_path, max_size=10, smooth_radius=0)
