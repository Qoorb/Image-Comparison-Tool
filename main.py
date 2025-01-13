import argparse
from typing import Dict
import numpy as np

from src.deep_learning import ImageSimilarityAnalyzer
from src.hashing import ImageHashAnalyzer
from src.utils import SimilarityVisualizer
from src.feature_matching import SIFTMatcher, ORBMatcher
from src.histogram import HistogramAnalyzer


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Image comparison using various methods"
    )
    parser.add_argument(
        "--image1", type=str, required=True, help="Path to the first image"
    )
    parser.add_argument(
        "--image2", type=str, required=True, help="Path to the second image"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["deep", "hash", "sift", "orb", "hist"],
        default="deep",
        help="Comparison method to use",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to pre-trained model (for deep learning method)",
    )
    parser.add_argument(
        "--hash_size",
        type=int,
        default=8,
        help="Hash size for perceptual hashing methods",
    )
    parser.add_argument(
        "--no_visualization",
        action="store_true",
        help="Disable results visualization"
    )

    return parser.parse_args()


def compute_deep_learning_similarity(
    image1_path: str, image2_path: str, model_path: str | None = None
) -> Dict[str, float]:
    analyzer = ImageSimilarityAnalyzer(model_path=model_path)
    similarity = analyzer.compute_similarity(image1_path, image2_path)
    return {"Siamese Network": similarity}


def compute_hash_similarity(
    image1_path: str, image2_path: str, hash_size: int = 8
) -> Dict[str, tuple[float, tuple[np.ndarray, np.ndarray]]]:
    analyzer = ImageHashAnalyzer(hash_size=hash_size)
    return analyzer.compare_images(image1_path, image2_path)


def compute_feature_matching_similarity(
    image1_path: str,
    image2_path: str,
    method: str = "sift",
    visualize: bool = False
) -> Dict[str, float]:
    matchers = {"sift": SIFTMatcher(), "orb": ORBMatcher()}
    matcher = matchers[method]

    keypoints1, keypoints2, good_matches = matcher.compare_images(
        image1_path, image2_path
    )

    if not keypoints1 or not keypoints2:
        return {f"{method.upper()} Matching": 0.0}

    similarity = len(good_matches) / min(len(keypoints1), len(keypoints2))
    result = {f"{method.upper()} Matching": similarity}

    if visualize:
        visualizer = SimilarityVisualizer()
        visualizer.visualize_comparison(
            image1_path,
            image2_path,
            result,
            keypoints1,
            keypoints2,
            good_matches
        )

    return result


def compute_histogram_similarity(
    image1_path: str, image2_path: str
) -> Dict[str, float]:
    analyzer = HistogramAnalyzer()
    return analyzer.compare_images(image1_path, image2_path)


def main() -> None:
    args = parse_arguments()
    similarities: Dict[
        str,
        float | tuple[float, tuple[np.ndarray, np.ndarray]]
    ] = {}

    if args.method == "deep":
        similarities.update(
            compute_deep_learning_similarity(
                args.image1,
                args.image2,
                args.model_path
            )
        )

    if args.method == "hash":
        similarities.update(
            compute_hash_similarity(args.image1, args.image2, args.hash_size)
        )

    if args.method == "sift":
        similarities.update(
            compute_feature_matching_similarity(
                args.image1,
                args.image2,
                "sift",
                visualize=not args.no_visualization
            )
        )

    if args.method == "orb":
        similarities.update(
            compute_feature_matching_similarity(
                args.image1,
                args.image2,
                "orb",
                visualize=not args.no_visualization
            )
        )

    if args.method == "hist":
        similarities.update(
            compute_histogram_similarity(args.image1, args.image2)
        )

    print("\nSimilarity scores:")
    for method, value in similarities.items():
        if isinstance(value, tuple):
            # for hash
            score, _ = value
            print(f"{method}: {score:.4f}")
        else:
            # for other methods
            print(f"{method}: {value:.4f}")

    if not args.no_visualization and args.method not in ["sift", "orb"]:
        visualizer = SimilarityVisualizer()
        visualizer.visualize_comparison(args.image1, args.image2, similarities)


if __name__ == "__main__":
    main()
