import torch
from vis.visulizeGrouped import visualize_with_trimesh

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

    distance = torch.cdist(centroids, x)

    idx = torch.argsort(distance, dim = -1)[:, :, :nsamples]

    grouped_x = index_point(x, idx)
    grouped_points = index_point(points, idx)
    labels = torch.argmin(distance, dim=1)
    # print(grouped_points.shape, labels.shape)
    # for i in range(x.shape[0]):
    #     visualize_with_trimesh(x[i].reshape(-1, 3), labels[i].reshape(-1))
    #     # print(grouped_x[i].reshape(-1, 3).shape, labels[i][:grouped_x[i].reshape(-1, 3).shape[0]].shape)
    #     # print(grouped_x[i].reshape(-1, 3).shape)
    #     # visualize_with_trimesh(grouped_x[i].reshape(-1, 3), labels[i][:grouped_x[i].reshape(-1, 3).shape[0]])
    
    return grouped_x, grouped_points, labels, idx