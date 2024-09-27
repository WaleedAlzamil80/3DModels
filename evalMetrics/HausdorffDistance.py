import torch
import torch.nn as nn

class HausdorffLoss(nn.Module):
    def __init__(self, alpha=10):
        """
        Soft approximation of the Hausdorff Distance using log-sum-exp.
        The parameter `alpha` controls the degree of approximation (higher is closer to max).
        """
        super(HausdorffLoss, self).__init__()
        self.alpha = alpha

    def forward(self, p1, p2):
        """
        Compute the Soft Hausdorff Distance between two point clouds.
        Args:
            p1 (torch.Tensor): Point cloud of shape (B, N, 3).
            p2 (torch.Tensor): Point cloud of shape (B, M, 3).
        Returns:
            torch.Tensor: Soft Hausdorff Distance as a loss value.
        """
        B, N, _ = p1.shape
        _, M, _ = p2.shape
        
        # Compute pairwise squared distances
        diff = p1.unsqueeze(2) - p2.unsqueeze(1)  # Shape (B, N, M, 3)
        dist = torch.sum(diff ** 2, dim=-1)  # Squared distances of shape (B, N, M)
        
        # Soft approximation to the maximum distance
        p1_to_p2 = torch.logsumexp(self.alpha * torch.min(dist, dim=-1)[0], dim=-1) / self.alpha
        p2_to_p1 = torch.logsumexp(self.alpha * torch.min(dist, dim=-2)[0], dim=-1) / self.alpha
        
        # Return the average of both directions
        hausdorff_dist = (p1_to_p2.mean() + p2_to_p1.mean()) / 2
        return hausdorff_dist