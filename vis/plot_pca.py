import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

def plot_pca(points, labels=None):
    """
    Apply PCA on a given n x d tensor and reduce it to n x 2.
    Optionally plot the results with different colors for each class based on labels.
    
    Args:
        points (torch.Tensor): Input tensor of shape (n, d).
        labels (torch.Tensor or np.ndarray, optional): Labels for each data point, used to color different classes.
    """

    # Convert points to numpy if it's a torch tensor
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()

    # Apply PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(points)
    
    # Plot results
    plt.figure(figsize=(8, 6))
    
    if labels is not None:
        # Convert labels to numpy if it's a torch tensor
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        # Get unique labels for different classes
        unique_labels = np.unique(labels)
        
        # Plot each class with a different color
        for label in unique_labels:
            idx = labels == label
            plt.scatter(reduced_data[idx, 0], reduced_data[idx, 1], label=f'Class {label}')
        plt.legend()
    else:
        # Plot without labels (all points in one color)
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1])

    plt.title('PCA Results (n x 2)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()