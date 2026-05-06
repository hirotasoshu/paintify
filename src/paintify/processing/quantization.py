from __future__ import annotations

from pathlib import Path

import numpy as np

from paintify.processing.color import rgb_to_lab
from paintify.processing.palette import CustomPalette, PaletteEntryBuilder


class KMeansQuantizer:
    color_bin_bits = 2

    def quantize(
        self,
        image: np.ndarray,
        max_colors: int,
        seed: int,
        palette_file: Path | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = image.shape[:2]
        quantized_rgb = self._quantized_rgb(image)
        unique_rgb, inverse, counts = np.unique(
            quantized_rgb.reshape(-1, 3),
            axis=0,
            return_inverse=True,
            return_counts=True,
        )
        custom_palette = CustomPalette.load(palette_file) if palette_file is not None else None
        effective_max_colors = max_colors
        if max_colors == -1:
            if custom_palette is None:
                raise ValueError("max_colors=-1 requires a palette file")
            effective_max_colors = custom_palette.color_count
        cluster_count = max(1, min(effective_max_colors, unique_rgb.shape[0]))
        if cluster_count == 1:
            centers = np.average(unique_rgb.astype(float), axis=0, weights=counts).reshape(1, 3)
        else:
            _, centers = self._deterministic_kmeans(
                unique_rgb.astype(float),
                counts.astype(float),
                cluster_count,
                seed,
            )

        centers_lab = rgb_to_lab(np.clip(centers, 0, 255).astype(np.uint8))
        if custom_palette is not None:
            centers_lab = custom_palette.snap_lab_colors(centers_lab)
        snapped_palette = PaletteEntryBuilder().build(centers_lab)

        lab_palette = np.array([entry.rgb for entry in snapped_palette], dtype=np.uint8)
        lab_palette = rgb_to_lab(lab_palette)
        unique_lab = rgb_to_lab(unique_rgb)
        distances = np.linalg.norm(unique_lab[:, None, :] - lab_palette[None, :, :], axis=2)
        palette_labels_by_unique_color = np.argmin(distances, axis=1).astype(np.int32)
        palette_labels = palette_labels_by_unique_color[inverse]
        return palette_labels.reshape(height, width), lab_palette

    def _quantized_rgb(self, image: np.ndarray) -> np.ndarray:
        return (image >> self.color_bin_bits << self.color_bin_bits).astype(np.uint8)

    def _deterministic_kmeans(
        self,
        points: np.ndarray,
        weights: np.ndarray,
        cluster_count: int,
        seed: int,
        iterations: int = 25,
    ) -> tuple[np.ndarray, np.ndarray]:
        centers = self._initial_centers(points, cluster_count, seed)
        labels = np.zeros(points.shape[0], dtype=np.int32)
        for _ in range(iterations):
            distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
            next_labels = np.argmin(distances, axis=1).astype(np.int32)
            if np.array_equal(next_labels, labels):
                break
            labels = next_labels
            for index in range(centers.shape[0]):
                member_mask = labels == index
                members = points[member_mask]
                if members.size > 0:
                    centers[index] = np.average(members, axis=0, weights=weights[member_mask])
        return labels, centers

    @staticmethod
    def _initial_centers(points: np.ndarray, cluster_count: int, seed: int) -> np.ndarray:
        unique_pixels = np.unique(points, axis=0)
        rng = np.random.default_rng(seed)
        if unique_pixels.shape[0] <= cluster_count:
            return unique_pixels.astype(float)
        choices = rng.choice(unique_pixels.shape[0], size=cluster_count, replace=False)
        return unique_pixels[np.sort(choices)].astype(float)
