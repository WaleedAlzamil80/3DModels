from scipy.spatial import cKDTree
import torch
import faiss

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


def kdneigh(x, args):

    if len(x.shape) == 2:
        x = x.unsqueeze(0)

    k = args.knn
    batch_size, num_points, num_features = x.shape

    edge_features = []

    for batch_idx in range(batch_size):
        # Convert tensor to numpy array for KDTree
        x_batch = x[batch_idx].detach().cpu().numpy()

        # Build a KDTree for efficient nearest neighbor search
        tree = cKDTree(x_batch)

        # Query the tree to get k-nearest neighbors (ignoring the point itself)
        _, idx = tree.query(x_batch, k=k+1)  # query k+1 to exclude self-neighbor
        idx = idx[:, 1:]  # Remove self-neighbor

        # Get the neighbors
        neighbors = torch.from_numpy(x_batch[idx]).to(x.device)

        # Compute edge features as concatenation of the central point and the difference with its neighbors
        central_point = x[batch_idx].unsqueeze(1).expand(-1, k, -1)
        edge_feature = torch.cat([central_point, neighbors - central_point], dim=-1)
        edge_features.append(edge_feature)

    edge_features = torch.stack(edge_features)
    
    return edge_features


def kdneighGPU(x, args):
    if len(x.shape) == 2:
        x = x.unsqueeze(0)  # Ensure we have a batch dimension

    k = args.knn
    batch_size, num_points, num_features = x.shape

    edge_features = []

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

    # Stack the edge features for the entire batch
    edge_features = torch.stack(edge_features)
    
    return edge_features


def neigh_faiss(x, args):

    if len(x.shape) == 2:
        x = x.unsqueeze(0)

    k = args.knn
    batch_size, num_points, num_features = x.shape

    edge_features = []

    for batch_idx in range(batch_size):
        # FAISS works on float32, ensure the data is of correct type
        x_batch = x[batch_idx].contiguous().float()

        # Build FAISS index for GPU
        index = faiss.IndexFlatL2(num_features)  # L2 distance
        res = faiss.StandardGpuResources()  # Create GPU resource
        index_gpu = faiss.index_cpu_to_gpu(res, 0, index)  # Move index to GPU

        # Add points to the index
        index_gpu.add(x_batch)

        # Perform k-NN search on GPU (excluding self, so k+1 neighbors)
        distances, idx = index_gpu.search(x_batch, k + 1)
        idx = idx[:, 1:]  # Exclude the first neighbor (self)

        # Gather neighbors using indices
        neighbors = x[batch_idx][idx]

        # Compute edge features as concatenation of central point and neighbors
        central_point = x[batch_idx].unsqueeze(1).expand(-1, k, -1)
        edge_feature = torch.cat([central_point, neighbors - central_point], dim=-1)
        edge_features.append(edge_feature)

    edge_features = torch.stack(edge_features)
    
    return edge_features