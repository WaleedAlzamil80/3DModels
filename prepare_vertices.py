import torch
import numpy as np
from factories.sampling_factory import get_sampling_technique

def preprocess(vertices_np, cetroids=4096, knn= 32, alpha = 2.0, delta = 5, sample=True, clean=True, labels=None):
    """
    Clean and sample the vertices.

    Args:
        vertices_np (np.ndarray): Input vertices of shape (num_points, 3).
        sample (bool): Whether to downsample the vertices using a sampling technique.
        clean (bool): Whether to clean the vertices by removing outliers.
        labels (np.ndarray): Input classes of shape (num_points, 1).

    Returns:
        np.ndarray: prepared points (num_points, 3).
        np.ndarray: Segmentation output of shape (num_points, ).
    """

    # Clean the input vertices to remove outliers
    if clean:
        origin = np.mean(vertices_np, axis=0)

        z_values = vertices_np[:, 2]
        y_values = vertices_np[:, 1]
        x_values = vertices_np[:, 0]

        y_mean, y_std = np.mean(y_values), np.std(y_values)
        x_mean, x_std = np.mean(x_values), np.std(x_values)

        valid_mask = (
            (z_values > (origin[2] - delta)) &
            (y_values < (y_mean + alpha * y_std)) & (y_values > (y_mean - alpha * y_std)) &
            (x_values < (x_mean + alpha * x_std)) & (x_values > (x_mean - alpha * x_std))
        )
        vertices_np = vertices_np[valid_mask]

    # Downsample the vertices if sampling is enabled
    if sample:
        sampling = get_sampling_technique("fpsample")
        vertices_np, idx = sampling(vertices_np, cetroids, knn)

    # If you have GT, make compartion 
    if clean and sample and labels is not None:
        labels = torch.tensor(np.array(labels, dtype=np.int64)[valid_mask][idx], dtype=torch.long).view(1, -1)
    elif sample and labels is not None:
        labels = torch.tensor(np.array(labels, dtype=np.int64)[idx], dtype=torch.long).view(1, -1)        
    elif clean and labels is not None:
        labels = torch.tensor(np.array(labels, dtype=np.int64)[valid_mask], dtype=torch.long).view(1, -1)
    elif labels is not None:
        labels = torch.tensor(np.array(labels, dtype=np.int64), dtype=torch.long).view(1, -1)

    return torch.tensor(vertices_np).view(1, -1, 3), labels
