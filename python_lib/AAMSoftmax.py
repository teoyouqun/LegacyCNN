import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AAMSoftmax(nn.Module):
    def __init__(self, in_feats, n_classes, margin=0.2, scale=30, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(n_classes, in_feats))
        self.device = device
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels):
        # Normalize features and weights
        embeddings = F.normalize(embeddings.to(self.device), p=2, dim=1)
        weight = F.normalize(self.weight.to(self.device), p=2, dim=1)

        cosine = F.linear(embeddings, weight)  # Cosine similarity [B, C]
        sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0.0, 1.0))

        # Add angular margin to target class
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot encode labels
        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).float()

        # Apply AAM
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.scale

        return logits  # Feed into CrossEntropyLoss