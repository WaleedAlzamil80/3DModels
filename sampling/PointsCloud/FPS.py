import torch 

device = "cuda" if torch.cuda.is_available() else "cpu"

def FPS(x, npoints):
    """
    Iterative Farthest Point Sampling (FPS) with vectorized distance calculations and caching.
    
    Args:
        x (torch.Tensor): Input point cloud coordinates of shape (B, N, 3), where B is batch size, N is the number of points, and 3 is the xyz coordinates.
        npoint (int): Number of points to sample.

    Returns:
        torch.Tensor: Indices of the sampled points, shape (B, npoint).
    """
    B, N, D = x.shape
    centroids = torch.zeros(B, npoints, dtype=torch.long).to(x.device)
    distance = torch.ones(B, N).to(x.device) + 1e10
    farthest = torch.randint(N, (B,), dtype=torch.long).to(x.device)
    batch_indices = torch.arange(B, dtype=torch.long).to(x.device)

    for i in range(npoints):
        centroids[:, i] = farthest
        centroid = x[batch_indices, farthest].view(B, 1, D)
        
        dist = torch.sum((x - centroid)**2, dim = -1)

        distance = torch.min(dist, distance)
        farthest = torch.max(distance, dim = -1)[1]

    return centroids