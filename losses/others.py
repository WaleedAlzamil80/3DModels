import torch
from torch import nn

def dice_loss(preds, labels, smooth=1):
    # Flatten predictions and labels
    preds = preds.contiguous().view(-1)
    labels = labels.contiguous().view(-1)

    # Calculate intersection and union
    intersection = (preds * labels).sum()
    union = preds.sum() + labels.sum()

    # Dice Loss
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, preds, labels):
        ce_loss = nn.CrossEntropyLoss()(preds, labels)
        pt = torch.exp(-ce_loss)  # pt is the probability of the correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def combined_loss(preds, labels):
    ce_loss = nn.CrossEntropyLoss()(preds, labels)
    d_loss = dice_loss(preds, labels)
    return ce_loss + d_loss
