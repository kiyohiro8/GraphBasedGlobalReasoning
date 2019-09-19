
import torch
from torch.nn import Module
import torch.nn.functional as F



class FocalLoss(Module):
    def __init__(self, weight=None, eps=1e-8, gamma=2):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, input, target):
        prob = F.softmax(input, dim=1)
        prob = torch.gather(prob, 1, torch.unsqueeze(target, dim=1))
        prob = torch.squeeze(prob, dim=1)
        loss = (1 - prob) ** self.gamma * F.cross_entropy(input, target, weight=self.weight, reduction="none")
        return torch.mean(loss)