from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError


class ImageInputError(ValueError):
    pass


class PillowImageLoader:
    def load(self, path: Path, max_size: int, smooth_radius: float) -> np.ndarray:
        try:
            with Image.open(path) as raw_image:
                image = ImageOps.exif_transpose(raw_image).convert("RGB")
        except (OSError, UnidentifiedImageError) as error:
            raise ImageInputError(f"{path} is not a readable image file") from error

        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        if smooth_radius > 0:
            image = image.filter(ImageFilter.GaussianBlur(radius=smooth_radius))
        return np.asarray(image, dtype=np.uint8)
