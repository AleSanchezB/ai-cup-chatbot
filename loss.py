import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.loss = nn.TripletMarginLoss(margin=margin, p=2)  # Distancia Euclidiana

    def forward(self, anchor, positive, negative):
        return self.loss(anchor, positive, negative)
