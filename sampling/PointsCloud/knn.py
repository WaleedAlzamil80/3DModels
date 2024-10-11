import torch

def pairwise_dist_blockwise(x, block_size=1024):
    """Compute pairwise distances in blocks to reduce memory usage."""
    batch_size, num_points, num_features = x.shape
    dist = torch.empty((batch_size, num_points, num_points), device=x.device)

    # Loop over blocks to compute pairwise distances
    for i in range(0, num_points, block_size):
        end_i = min(i + block_size, num_points)
        for j in range(0, num_points, block_size):
            end_j = min(j + block_size, num_points)
            dist[:, i:end_i, j:end_j] = torch.cdist(x[:, i:end_i], x[:, j:end_j], p=2)

    return dist

def neigh(x, args):

    if len(x.shape) == 2:
        x = x.unsqueeze(0)

    k = args.knn
    batch_size, num_points, num_features = x.shape

    # Compute pairwise distances
    dist = torch.cdist(x, x, p=2)
    # dist = pairwise_dist_blockwise(x)

    # Mask the diagonal to avoid including the point itself as a neighbor
    dist.masked_fill_(torch.eye(num_points, device=x.device).bool().unsqueeze(0), float('inf'))  # Set self-distances to infinity
    # for i in range(batch_size):
    #     dist[i].fill_diagonal_(float('inf'))

    # Get the indices of the k-nearest neighbors
    _, idx = dist.topk(k, dim=-1, largest=False)

    # Efficient neighbor gathering using advanced indexing (avoids expansion)
    neighbors = x[torch.arange(batch_size).unsqueeze(1).unsqueeze(2), idx]

    # neighbors = torch.gather(x.unsqueeze(1).expand(-1, num_points, -1, -1), 2, idx.unsqueeze(3).expand(-1, -1, -1, num_features))

    # Compute edge features
    edge_features = torch.cat([x.unsqueeze(2).expand(-1, -1, k, -1), neighbors - x.unsqueeze(2)], dim=-1)

    return edge_features