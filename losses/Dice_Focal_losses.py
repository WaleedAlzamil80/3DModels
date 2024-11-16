import torch
from torch import nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, preds, labels):
        ce_loss = nn.CrossEntropyLoss()(preds, labels)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth = 1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, labels):
        # Flatten predictions and labels
        preds = preds.contiguous().view(-1)
        labels = labels.contiguous().view(-1)

        # Calculate intersection and union
        intersection = (preds * labels).sum()
        union = preds.sum() + labels.sum()

        # Dice Loss
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


def combined_loss(preds, labels):
    ce_loss = nn.CrossEntropyLoss()(preds, labels)
    d_loss = DiceLoss()(preds, labels)
    return ce_loss + d_loss
