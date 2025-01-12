from typing import Dict, List
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np


class SimilarityVisualizer:
    def __init__(self, figsize: tuple[int, int] = (15, 5)):
        self.figsize = figsize

    def visualize_comparison(
        self,
        image1_path: str,
        image2_path: str,
        similarities: Dict[str, float],
        keypoints1: List | None = None,
        keypoints2: List | None = None,
        good_matches: List | None = None,
    ) -> None:
        if keypoints1 and keypoints2 and good_matches:
            self._plot_feature_matching(
                image1_path, image2_path, keypoints1, keypoints2, good_matches
            )
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=self.figsize)
            self._plot_images(ax1, ax2, image1_path, image2_path)
            self._plot_similarities(ax3, similarities)
            plt.tight_layout()
            plt.show()

    def _plot_feature_matching(
        self,
        image1_path: str,
        image2_path: str,
        keypoints1: List,
        keypoints2: List,
        good_matches: List,
    ) -> None:
        img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

        img_matches = cv2.drawMatches(
            img1,
            keypoints1,
            img2,
            keypoints2,
            good_matches,
            outImg=np.array([]),
            matchesThickness=1,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        plt.figure(figsize=self.figsize)
        plt.imshow(img_matches)
        plt.title(f"Number of good matches: {len(good_matches)}")
        plt.axis("off")
        plt.show()

    def _plot_images(
        self,
        ax1: plt.Axes,
        ax2: plt.Axes,
        image1_path: str,
        image2_path: str
    ) -> None:
        img1 = Image.open(image1_path)
        img2 = Image.open(image2_path)

        ax1.imshow(img1)
        ax1.set_title("Image 1")
        ax1.axis("off")

        ax2.imshow(img2)
        ax2.set_title("Image 2")
        ax2.axis("off")

    def _plot_similarities(
            self,
            ax: plt.Axes,
            similarities: Dict[str, float]
    ) -> None:
        methods = list(similarities.keys())
        scores = list(similarities.values())

        ax.bar(methods, scores)
        ax.set_title("Similarity Scores\n")
        plt.setp(ax.get_xticklabels(), rotation=45)
