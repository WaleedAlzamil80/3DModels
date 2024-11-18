import torch
import torch.nn as nn

class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()
 
    def forward(self, p1, p2):
        """
        Compute the Chamfer Distance between two point clouds.
        Args:
            p1 (torch.Tensor): Point cloud of shape (B, N, 3).
            p2 (torch.Tensor): Point cloud of shape (B, M, 3).
        Returns:
            torch.Tensor: Chamfer Distance as a loss value.
        """
        B, N, _ = p1.shape
        _, M, _ = p2.shape

        # Compute pairwise squared distances
        diff = p1.unsqueeze(2) - p2.unsqueeze(1)  # Shape (B, N, M, 3)
        dist = torch.sum(diff ** 2, dim=-1)  # Squared distances of shape (B, N, M)

        # For each point in p1, find the nearest point in p2
        p1_to_p2 = torch.min(dist, dim=-1)[0]  # Shape (B, N)

        # For each point in p2, find the nearest point in p1
        p2_to_p1 = torch.min(dist, dim=-2)[0]  # Shape (B, M)

        # Average over all points
        chamfer_dist = (p1_to_p2.mean(dim=-1) + p2_to_p1.mean(dim=-1)).mean()

        return chamfer_dist


class DistanceDk(nn.Module):
    def __init__(self):
        super(DistanceDk, self).__init__()
 
    def forward(self, p1, p2):
        """
        Compute the Squared Distance between two point clouds.
        Args:
            p1 (torch.Tensor): Point cloud of shape (B, N, 3).
            p2 (torch.Tensor): Point cloud of shape (B, N, 3).
        Returns:
            torch.Tensor: Squared Distance as a loss value.
        """

        return torch.sum((p1 - p2) ** 2, dim=-1).mean(dim=-1).mean()


class chamferDk(nn.Module):
    def __init__(self, alpha = 0.5):
        super(chamferDk, self).__init__()
        self.l2 = DistanceDk()
        self.chamfer = ChamferLoss()
        self.alpha = alpha

    def forward(self, preds, labels):
        return self.alpha * self.chamfer(preds, labels) + (1 - self.alpha) * self.l2(preds, labels)
