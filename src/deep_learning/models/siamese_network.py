import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights  # type: ignore


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SiameseNetwork, self).__init__()

        self.base_network = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.base_network = nn.Sequential(
            *list(self.base_network.children())[:-1]
        )
        self.embedding = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, embedding_dim)
        )

    def forward_one(self, x):
        x = self.base_network(x)
        x = x.view(x.size()[0], -1)
        x = self.embedding(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2
