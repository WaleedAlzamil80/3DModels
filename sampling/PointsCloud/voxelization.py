import open3d as o3d
import numpy as np
from .fpsample import fps as FPS
from .fpsample import fpsample_fps as FPSample

def voxel_grid_downsampling(point_cloud, voxel_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    return np.asarray(downsampled_pcd.points)
 
def downsample_to_fixed_vertices(vertices, num_target_points, k, voxel_size=0.5):
    """
    Downsample point cloud to a fixed number of vertices.

    Parameters:
    - vertices: (N, 3) array of input points.
    - num_target_points: Number of points desired after downsampling.
    - voxel_size: Voxel size used for the initial downsampling.

    Returns:
    - downsampled_vertices: (num_target_points, 3) array of downsampled points.
    """
    # Initial downsampling using voxel grid (optional)
    downsampled_vertices = voxel_grid_downsampling(vertices, voxel_size)
    # print(f"After voxel downsampling: {downsampled_vertices.shape[0]} points")

    # If the downsampled points are greater than the target, apply FPS to reduce
    if downsampled_vertices.shape[0] > num_target_points:
        # downsampled_vertices, centroids_idx = FPS(downsampled_vertices, num_target_points*k)
        downsampled_vertices, centroids_idx = FPSample(downsampled_vertices, num_target_points, k)

    # If the downsampled points are fewer than the target, do random sampling
    elif downsampled_vertices.shape[0] < num_target_points:
        indices = np.random.choice(downsampled_vertices.shape[0], num_target_points, replace=True)
        downsampled_vertices = downsampled_vertices[indices]

    return downsampled_vertices
