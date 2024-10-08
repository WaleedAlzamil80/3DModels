import torch

# note for our data: High number of centroids is better from having high number of smaples
## while we go deeper we can reduce the number of centroids and incread the number of samples
 
def Grouping(x, points, centroids, nsamples, radius):
    """
    Optimized grouping of nearby points for each centroid with efficient neighbor search and vectorized operations.

    Args:
        x (torch.Tensor): Point cloud coordinates of shape (B, N, 3).
        points (torch.Tensor): Additional feature vectors of shape (B, N, D) for each point.
        centroids (torch.Tensor): Centroids of shape (B, C, 3).
        nsample (int): Number of nearest points to sample for each centroid.
        radius (float): Maximum distance to consider as neighbors (ball query, I didn't use it as I want a fixed number of point for vectorization).

    Returns:
        torch.Tensor: Grouped coordinates of shape (B, C, nsample, 3).
        torch.Tensor: Grouped feature vectors of shape (B, C, nsample, D).
    """
    if len(x.shape) == 2:
        x = x.unsqueeze(0)

    distance = torch.cdist(centroids, x)

    idx = torch.argsort(distance, dim = -1)[:, :, :nsamples]

    grouped_x = index_point(x, idx)
    grouped_points = index_point(points, idx)
    labels = torch.argmin(distance, dim=1)

    return grouped_x, grouped_points, labels, idx


def index_point(x, idx):
    """
    Indexes the points according to provided indices.

    Args:
        x (torch.Tensor): Point cloud or feature vectors of shape (B, N, D).
        idx (torch.Tensor): Indices of shape (B, C, nsample) or (B, C) to gather from points.

    Returns:
        torch.Tensor: Gathered points of shape (B, C, nsample, D).
    """

    if len(idx.shape) == 3:
        return x[torch.arange(x.shape[0], device=x.device).view(x.shape[0], 1, 1).expand(-1, idx.shape[1], idx.shape[2]), idx]
    else:
        return x[torch.arange(x.shape[0], device=x.device).view(x.shape[0], 1).expand(-1, idx.shape[1]), idx]