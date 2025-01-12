from abc import ABC, abstractmethod
import cv2
import numpy as np
from typing import Tuple, List, Any


class FeatureMatcher(ABC):
    @abstractmethod
    def detect_and_compute(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        pass

    @abstractmethod
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List:
        pass

    def compare_images(
        self, image1_path: str, image2_path: str, threshold: float = 0.7
    ) -> Tuple[List, List, List]:
        img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

        keypoints1, descriptors1 = self.detect_and_compute(img1)
        keypoints2, descriptors2 = self.detect_and_compute(img2)

        if descriptors1 is None or descriptors2 is None:
            return [], [], []

        matches = self.match_features(descriptors1, descriptors2)
        good_matches = self.filter_matches(matches, threshold)

        return keypoints1, keypoints2, good_matches

    @staticmethod
    def filter_matches(matches: List, threshold: float) -> List:
        good_matches = []
        for m, n in matches:
            if m.distance < threshold * n.distance:
                good_matches.append(m)
        return good_matches


class SIFTMatcher(FeatureMatcher):
    def __init__(self):
        self.detector = cv2.SIFT_create()
        self.matcher = self._create_matcher()

    def detect_and_compute(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        return self.detector.detectAndCompute(image, None)

    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List:
        return self.matcher.knnMatch(desc1, desc2, k=2)

    @staticmethod
    def _create_matcher():
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        return cv2.FlannBasedMatcher(index_params, search_params)


class ORBMatcher(FeatureMatcher):
    def __init__(self, n_features: int = 500):
        self.detector = cv2.ORB_create(nfeatures=n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def detect_and_compute(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        return self.detector.detectAndCompute(image, None)

    def match_features(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray
    ) -> Any:
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        return matches
