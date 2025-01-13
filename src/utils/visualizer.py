from typing import Dict, List, Union
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
        similarities: Dict[
            str, Union[float, tuple[float, tuple[np.ndarray, np.ndarray]]]
        ],
        keypoints1: List | None = None,
        keypoints2: List | None = None,
        good_matches: List | None = None,
    ) -> None:
        if keypoints1 and keypoints2 and good_matches:
            self._plot_feature_matching(
                image1_path, image2_path, keypoints1, keypoints2, good_matches
            )
        else:
            hash_similarities = {}
            other_similarities = {}

            for method, value in similarities.items():
                if isinstance(value, tuple):
                    hash_similarities[method] = value
                else:
                    other_similarities[method] = value

            if hash_similarities:
                fig = plt.figure(figsize=(15, 10))
                ax1 = plt.subplot2grid((2, 2), (0, 0))
                ax2 = plt.subplot2grid((2, 2), (0, 1))
                self._plot_images(ax1, ax2, image1_path, image2_path)

                ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
                self._plot_hashes(ax3, hash_similarities)
            else:
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=self.figsize)
                self._plot_images(ax1, ax2, image1_path, image2_path)
                self._plot_similarities(ax3, other_similarities)

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
        self, ax1: plt.Axes, ax2: plt.Axes, image1_path: str, image2_path: str
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

        bars = ax.bar(methods, scores)
        ax.set_title("Similarity Scores\n")
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.set_ylim((0, 1))
        # plt.tight_layout()

        for bar in bars:
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                yval, f"{yval:.2f}",
                ha='center',
                va='bottom'
            )

    def _plot_hashes(
        self,
        ax: plt.Axes,
        hash_similarities: Dict[
            str,
            tuple[float, tuple[np.ndarray, np.ndarray]]
        ]
    ) -> None:
        current_pos = 0

        for method, (score, (hash1, hash2)) in hash_similarities.items():
            hash1_img = hash1.reshape((8, 8))
            hash2_img = hash2.reshape((8, 8))

            ax_hash1 = ax.inset_axes(
                [0.3, 1.0 - (current_pos + 1) * 0.25, 0.15, 0.2]
            )
            ax_hash2 = ax.inset_axes(
                [0.55, 1.0 - (current_pos + 1) * 0.25, 0.15, 0.2]
            )

            ax_hash1.imshow(hash1_img, cmap="binary")
            ax_hash2.imshow(hash2_img, cmap="binary")

            ax_hash1.axis("off")
            ax_hash2.axis("off")

            ax.text(
                0.5,
                1.0 - current_pos * 0.25 - 0.1,
                f"{method}\nScore: {score:.4f}",
                ha="center",
                va="bottom",
            )

            current_pos += 1

        ax.axis("off")
        ax.set_title("Hash Visualizations")
