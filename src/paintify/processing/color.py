import cv2
import numpy as np


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    normalized = rgb.reshape(1, -1, 3).astype(np.float32) / 255.0
    return cv2.cvtColor(normalized, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(lab.reshape(1, -1, 3).astype(np.float32), cv2.COLOR_LAB2RGB).reshape(-1, 3)
    return np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8)
