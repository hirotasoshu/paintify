from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class ImageInputError(ValueError):
    pass


class OpenCvImageLoader:
    def load(self, path: Path, max_size: int, smooth_radius: float) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ImageInputError(f"{path} is not a readable image file")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        scale = min(max_size / width, max_size / height, 1.0)
        if scale < 1.0:
            image = cv2.resize(
                image,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        if smooth_radius > 0:
            image = cv2.GaussianBlur(image, (0, 0), smooth_radius)
        return image.astype(np.uint8, copy=False)
