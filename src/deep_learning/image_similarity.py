import torch
import torch.nn as nn

from .models import SiameseNetwork
from .utils import ImagePreprocessor


class ImageSimilarityAnalyzer:
    def __init__(self, model_path: str | None = None):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self._initialize_model(model_path)
        self.preprocessor = ImagePreprocessor()
        self.similarity_fn = nn.CosineSimilarity(dim=1)

    def _initialize_model(
        self,
        model_path: str | None = None
    ) -> SiameseNetwork:
        model = SiameseNetwork().to(self.device)
        if model_path:
            model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    def compute_similarity(self, image1_path: str, image2_path: str) -> float:
        with torch.no_grad():
            img1 = self.preprocessor.preprocess(image1_path)
            img2 = self.preprocessor.preprocess(image2_path)

            embedding1, embedding2 = self.model(img1, img2)
            similarity = self.similarity_fn(embedding1, embedding2).item()

            return similarity
