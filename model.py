# model.py
import torch
import torch.nn as nn
from torchvision import models

class SiameseNetwork(nn.Module):
    def __init__(self, base_model=None):
        super(SiameseNetwork, self).__init__()
        if base_model is None:
            base_model = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])  # Quitamos la última capa
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, img1, img2):
        feat1 = self.feature_extractor(img1)
        feat2 = self.feature_extractor(img2)
        feat1 = feat1.view(feat1.size(0), -1)
        feat2 = feat2.view(feat2.size(0), -1)
        distance = torch.abs(feat1 - feat2)
        output = self.fc(distance)
        output = torch.sigmoid(output)  # Aplicamos Sigmoid para restringir la salida en [0, 1]
        return output

class TripletSiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super(TripletSiameseNetwork, self).__init__()

        # Base de la red (puede ser cualquier CNN, aquí ResNet18)
        self.base_network = models.resnet18(pretrained=True)
        self.base_network.fc = nn.Linear(self.base_network.fc.in_features, embedding_dim)  # Capa final de embedding

    def forward(self, anchor, positive, negative):
        anchor_embedding = self.base_network(anchor)
        positive_embedding = self.base_network(positive)
        negative_embedding = self.base_network(negative)
        return anchor_embedding, positive_embedding, negative_embedding
