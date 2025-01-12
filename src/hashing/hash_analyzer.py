from typing import Dict, Union
import cv2
import numpy as np
from PIL import Image
import imagehash
from scipy.fftpack import dct  # type: ignore
from pathlib import Path


class ImageHashAnalyzer:
    def __init__(self, hash_size: int = 8):
        self.hash_size = hash_size

    def compute_average_hash(self, image_path: Union[str, Path]) -> np.ndarray:
        image = self._load_and_resize_image(
            image_path, (self.hash_size, self.hash_size)
        )
        mean = np.mean(image)
        return np.array(
            [1 if pixel >= mean else 0 for pixel in image.flatten()]
        )

    def compute_difference_hash(
            self,
            image_path: Union[str, Path]
    ) -> np.ndarray:
        image = self._load_and_resize_image(
            image_path, (self.hash_size + 1, self.hash_size)
        )
        return np.array(
            [
                1 if image[i, j] > image[i, j + 1] else 0
                for i in range(self.hash_size)
                for j in range(self.hash_size)
            ]
        )

    def compute_perceptual_hash(
            self,
            image_path: Union[str, Path]
    ) -> np.ndarray:
        image = self._load_and_resize_image(image_path, (32, 32))
        dct_result = dct(dct(image.astype(float), axis=0), axis=1)
        dct_low = dct_result[: self.hash_size, : self.hash_size]
        med = np.median(dct_low)
        return np.array([1 if val > med else 0 for val in dct_low.flatten()])

    def compute_wavelet_hash(
            self,
            image_path: Union[str, Path]
    ) -> imagehash.ImageHash:
        return imagehash.whash(Image.open(image_path))

    def _load_and_resize_image(
        self, image_path: Union[str, Path], size: tuple
    ) -> np.ndarray:
        return cv2.resize(
            cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE),
            size
        )

    def compare_images(
        self, image1_path: Union[str, Path], image2_path: Union[str, Path]
    ) -> Dict[str, float]:
        return {
            "Average Hash": self._hamming_distance(
                self.compute_average_hash(image1_path),
                self.compute_average_hash(image2_path),
            ),
            "Difference Hash": self._hamming_distance(
                self.compute_difference_hash(image1_path),
                self.compute_difference_hash(image2_path),
            ),
            "Perceptual Hash": self._hamming_distance(
                self.compute_perceptual_hash(image1_path),
                self.compute_perceptual_hash(image2_path),
            ),
            "Wavelet Hash": float(
                abs(
                    self.compute_wavelet_hash(image1_path)
                    - self.compute_wavelet_hash(image2_path)
                )
            ),
        }

    @staticmethod
    def _hamming_distance(hash1: np.ndarray, hash2: np.ndarray) -> float:
        return float(np.sum(hash1 != hash2))
