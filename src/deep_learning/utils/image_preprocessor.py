import torch
import torchvision.transforms as transforms  # type: ignore
from PIL import Image


class ImagePreprocessor:
    def __init__(self, image_size: tuple[int, int] = (224, 224)):
        self.transform = self._create_transform(image_size)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def _create_transform(
            self,
            image_size: tuple[int, int]
    ) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def preprocess(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image)
        return tensor.unsqueeze(0).to(self.device)
