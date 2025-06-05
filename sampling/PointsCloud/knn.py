import torch
from torch_cluster import knn_graph
# from torch_geometric.nn import knn_graph

def knn_neighbors(x, args):
    B, N, C = x.shape
    x_flat = x.reshape(B * N, C)
    k = args.knn

    batch = torch.arange(B, device=x.device).repeat_interleave(N)
    edge_index = knn_graph(x_flat, k=k, batch=batch, loop=False)

    src, dst = edge_index
    dst_batch = batch[dst]
    dst_idx_local = dst % N
    src_idx_local = src % N

    idx = torch.zeros((B, N, k), dtype=torch.long, device=x.device)
    arange_k = torch.arange(k, device=x.device).repeat(B * N)[:dst.size(0)]

    idx[dst_batch, dst_idx_local, arange_k] = src_idx_local

    neighbors = torch.gather(
        x.unsqueeze(2).expand(-1, -1, k, -1),
        dim=1,
        index=idx.unsqueeze(-1).expand(-1, -1, -1, C)
    )

    central = x.unsqueeze(2).expand(-1, -1, k, -1)
    edge_feats = torch.cat([central, neighbors - central], dim=-1)
    return edge_feats, neighbors

def compute_local_covariance(points):
    """
    Compute the local covariance matrix for each point cloud.

    Args:
        points: Tensor of shape [batch_size, num_points, k_nearest, 3].

    Returns:
        covariances: Tensor of shape [batch_size, num_points, 9].
    """
    # Calculate mean across neighbors (dim=-2)
    means = points.mean(dim=-2, keepdim=True)  # Shape: [batch_size, num_points, 1, 3]

    # Subtract mean from neighbors
    centered_points = points - means  # Shape: [batch_size, num_points, k_nearest, 3]

    # Compute outer product of centered points (batched matrix multiplication)
    # Reshape centered points for bmm
    centered_points_flat = centered_points.view(-1, points.size(-2), points.size(-1))  # Shape: [B*num_points, k_nearest, 3]
    cov_matrices = torch.bmm(centered_points_flat.transpose(1, 2), centered_points_flat)  # Shape: [B*num_points, 3, 3]

    # Normalize by the number of neighbors (k_nearest)
    cov_matrices /= points.size(-2)  # Normalize by K

    # Reshape back to batch structure and flatten the 3x3 matrix
    cov_matrices = cov_matrices.view(points.size(0), points.size(1), 9)  # Shape: [batch_size, num_points, 9]

    return cov_matrices
