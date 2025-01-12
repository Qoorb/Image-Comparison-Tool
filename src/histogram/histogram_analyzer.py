import cv2
import numpy as np
from typing import Dict


class HistogramAnalyzer:
    def __init__(self):
        self.comparison_methods = [
            ("Correlation", cv2.HISTCMP_CORREL),
            ("Chi-Square", cv2.HISTCMP_CHISQR),
            ("Intersection", cv2.HISTCMP_INTERSECT),
            ("Bhattacharyya", cv2.HISTCMP_BHATTACHARYYA),
        ]

    def compute_histogram(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def compare_images(
            self,
            image1_path: str,
            image2_path: str
    ) -> Dict[str, float]:
        hist1 = self.compute_histogram(image1_path)
        hist2 = self.compute_histogram(image2_path)

        results = {}
        for method_name, method in self.comparison_methods:
            score = cv2.compareHist(hist1, hist2, method)
            # Нормализация
            if method == cv2.HISTCMP_INTERSECT:
                score = score / (hist1.shape[0] * hist1.shape[1])
            elif method in [cv2.HISTCMP_CHISQR, cv2.HISTCMP_BHATTACHARYYA]:
                score = 1 - score
            results[f"Histogram {method_name}"] = score

        return results
