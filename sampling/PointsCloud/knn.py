from scipy.spatial import cKDTree
import torch
from torch_cluster import knn_graph
# from torch_geometric.nn import knn_graph

def knn_neighbors(x, args):
    """Corrected kNN implementation with torch_cluster"""
    batch_size, num_points, _ = x.shape
    x_flat = x.reshape(-1, x.size(-1))  # Shape: (B*N, D)
    k = args.knn

    # Create batch indices
    batch = torch.arange(batch_size, device=x.device)
    batch = batch.repeat_interleave(num_points)  # Shape: (B*N,)
    # Get kNN indices (returns edge index: [2, E]) (neighbors, query_points)
    edge_index = knn_graph(x_flat, k=k, batch=batch, loop=False, flow="source_to_target")

    # Initialize idx tensor to hold neighbors
    idx = edge_index[0].view(batch_size, num_points, k)
    print(idx.shape)
    #  local_idx = edge_index[1] - (batch * num_points)[edge_index[0]]
    #  local_idx = local_idx.view(batch_size, num_points, k)

    # Gather neighbors
    x_neighbors = x.unsqueeze(2).expand(-1, -1, k, -1)
    neighbors = torch.gather(
        x_neighbors,
        dim=1,
        index=idx.unsqueeze(-1).expand(-1, -1, -1, x.size(-1))
    )
    # Compute edge features
    central = x.unsqueeze(2).expand(-1, -1, k, -1)  # (B, N, k, C)

    edge_feats = torch.cat([central, neighbors - central], dim=-1)
    return edge_feats, neighbors

def kdneighGPU(x, args):
    if len(x.shape) == 2:
        x = x.unsqueeze(0)  # Ensure we have a batch dimension

    k = args.knn
    batch_size, num_points, num_features = x.shape

    edge_features = []
    neighborsF = []

    # Loop over the batch
    for batch_idx in range(batch_size):
        # Convert the tensor to a numpy array for KDTree computation (no gradients required for this step)
        x_batch_np = x[batch_idx].detach().cpu().numpy()

        # Build a KDTree for efficient nearest neighbor search
        tree = cKDTree(x_batch_np)

        # Query the KDTree for k-nearest neighbors (excluding self, so query k+1)
        _, idx = tree.query(x_batch_np, k=k + 1)
        idx = idx[:, 1:]  # Exclude self-neighbor by slicing off the first element

        # Gather neighbors based on the KDTree indices, but using PyTorch tensors
        neighbors = x[batch_idx][torch.tensor(idx, device=x.device, dtype=torch.long)]

        # Compute edge features as the concatenation of the central point and the difference with neighbors
        central_point = x[batch_idx].unsqueeze(1).expand(-1, k, -1)
        edge_feature = torch.cat([central_point, neighbors - central_point], dim=-1)

        edge_features.append(edge_feature)
        neighborsF.append(neighbors)

    # Stack the edge features for the entire batch
    edge_features = torch.stack(edge_features)
    neighborsF = torch.stack(neighborsF)

    return edge_features, neighborsF

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
