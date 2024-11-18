import torch
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, preds, labels):
        ce_loss = nn.CrossEntropyLoss()(preds, labels)

        # Compute softmax probabilities
        probs = F.softmax(preds, dim=1)
        labels_one_hot = F.one_hot(labels, num_classes=preds.size(1)).permute(0, 2, 1).float()

        # Gather probabilities for the true class
        pt = torch.sum(probs * labels_one_hot, dim=1)  # [batch_size, num_points]

        # Compute Cross-Entropy Loss
        ce_loss = F.cross_entropy(preds, labels, reduction='none')  # [batch_size, num_points]

        # Compute Focal Loss
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
