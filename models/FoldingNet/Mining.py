import torch
import torch.nn as nn 

class GaussianKernelConv(nn.Module):
    def __init__(self, in_features, num_kernel_points, sigma=1.0):
        super(GaussianKernelConv, self).__init__()

        # Define learnable kernel points and sigma
        self.kernel_points = nn.Parameter(torch.randn(num_kernel_points, in_features))
        self.sigma = nn.Parameter(torch.tensor(sigma))

    def forward(self, neighborhoods): # Shape: (batch_size, num_points, k, in_features)
        # Compute Gaussian correlation between neighborhoods and kernel points
        diff = neighborhoods.unsqueeze(3) - self.kernel_points
        distance_sq = torch.sum(diff ** 2, dim=-1)  # (batch_size, num_points, k, num_kernel_points)

        # Apply Gaussian function
        kernel_corr = torch.exp(-distance_sq / (2 * self.sigma ** 2))  # Shape: (batch_size, num_points, k, num_kernel_points)
 
        # Average over neighborhood points to summarize correlation
        kernel_corr = kernel_corr.mean(dim=2)  # (batch_size, num_points, num_kernel_points)

        return kernel_corr
