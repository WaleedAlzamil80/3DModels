import torch

def neigh(x, args):

    if len(x.shape) == 2:
        x = x.unsqueeze(0)

    k = args.knn
    batch_size, num_points, num_features = x.shape

    # Compute pairwise distances
    dist = torch.cdist(x, x, p=2)

    # Mask the diagonal to avoid including the point itself as a neighbor
    mask = torch.eye(num_points, device=x.device).bool().unsqueeze(0)
    dist.masked_fill_(mask, float('inf'))  # Set self-distances to infinity

    # Get the indices of the k-nearest neighbors
    _, idx = dist.topk(k, dim=-1, largest=False)

    neighbors = torch.gather(x.unsqueeze(1).expand(-1, num_points, -1, -1), 2, idx.unsqueeze(3).expand(-1, -1, -1, num_features))

    # Compute edge features
    edge_features = torch.cat([x.unsqueeze(2).expand(-1, -1, k, -1), neighbors - x.unsqueeze(2)], dim=-1)

    return edge_features