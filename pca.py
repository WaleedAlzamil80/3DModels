import torch

def pca(X, num_components):
    """
    Perform PCA on the input data X and return the projected data.

    Args:
        X (torch.Tensor): Input data of shape (n_samples, n_features)
        num_components (int): Number of principal components to keep

    Returns:
        X_pca (torch.Tensor): Data projected onto principal components (n_samples, num_components)
        components (torch.Tensor): Principal components (num_components, n_features)
        explained_variance (torch.Tensor): Eigenvalues (num_components,)
    """
    # Step 1: Center the data
    X_mean = X.mean(dim=0)
    X_centered = X - X_mean

    # Step 2: Compute the covariance matrix
    cov_matrix = torch.mm(X_centered.T, X_centered) / (X.shape[0] - 1)

    # Step 3: Eigen decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

    # Step 4: Sort eigenvalues and eigenvectors in descending order
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    top_eigenvalues = eigenvalues[sorted_indices][:num_components]
    top_eigenvectors = eigenvectors[:, sorted_indices][:, :num_components]

    # Step 5: Project the data
    X_pca = torch.mm(X_centered, top_eigenvectors)

    return X_pca, top_eigenvectors.T, top_eigenvalues


def batched_pca(X, num_components):
    """
    Batched PCA on input tensor X.

    Args:
        X (torch.Tensor): shape (B, N, D) = (batch_size, num_points, features)
        num_components (int): number of principal components to retain

    Returns:
        X_pca: shape (B, N, num_components)
        components: shape (B, num_components, D)
        explained_variance: shape (B, num_components)
    """
    B, N, D = X.shape

    # Step 1: Center the data
    mean = X.mean(dim=1, keepdim=True)  # shape: (B, 1, D)
    X_centered = X - mean  # shape: (B, N, D)

    # Step 2: Compute covariance matrices
    cov = torch.matmul(X_centered.transpose(1, 2), X_centered) / (N - 1)  # shape: (B, D, D)

    # Step 3: Eigen decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)  # shapes: (B, D), (B, D, D)

    # Step 4: Sort in descending order
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    eigenvalues_sorted = torch.gather(eigenvalues, 1, sorted_indices[:, :num_components])
    
    batch_indices = torch.arange(B).unsqueeze(1)
    top_eigenvectors = eigenvectors[batch_indices, :, sorted_indices[:, :num_components]]  # shape: (B, D, K)

    # Step 5: Project data
    X_pca = torch.matmul(X_centered, top_eigenvectors)  # shape: (B, N, K)

    return X_pca, top_eigenvectors.transpose(1, 2), eigenvalues_sorted
