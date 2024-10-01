import torch
import open3d as o3d
import numpy as np
from .FPS import FPS
from .Grouping import Grouping, index_point

def voxel_grid_downsampling(point_cloud, voxel_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    return np.asarray(downsampled_pcd.points)

def downsample_to_fixed_vertices(vertices, num_target_points, initial_downsampling=True, voxel_size=0.01):
    """
    Downsample point cloud to a fixed number of vertices.

    Parameters:
    - vertices: (N, 3) array of input points.
    - num_target_points: Number of points desired after downsampling.
    - initial_downsampling: Apply initial voxel grid downsampling or not.
    - voxel_size: Voxel size used for the initial downsampling.
    
    Returns:
    - downsampled_vertices: (num_target_points, 3) array of downsampled points.
    """
    # Initial downsampling using voxel grid (optional)
    if initial_downsampling:
        downsampled_vertices = voxel_grid_downsampling(vertices, voxel_size)
        print(f"After voxel downsampling: {downsampled_vertices.shape[0]} points")
    else:
        downsampled_vertices = vertices

    # If the downsampled points are greater than the target, apply FPS to reduce
    if downsampled_vertices.shape[0] > num_target_points:
        downsampled_vertices = torch.tensor(downsampled_vertices, dtype=torch.float32).unsqueeze(0)  # Add batch dim
        centroids_idx = FPS(downsampled_vertices, num_target_points)
        downsampled_vertices = index_point(downsampled_vertices, centroids_idx).squeeze(0).numpy()  # Remove batch dim
    # If the downsampled points are fewer than the target, do random sampling
    elif downsampled_vertices.shape[0] < num_target_points:
        indices = np.random.choice(downsampled_vertices.shape[0], num_target_points, replace=True)
        downsampled_vertices = downsampled_vertices[indices]

    return downsampled_vertices
