import torch
import numpy as np
import trimesh
import time
import argparse
from sampling.PointsCloud.FPS import FPS
from sampling.PointsCloud.Grouping import Grouping, index_point

# Define the argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark FPS and Grouping on 3D vertices")
    
    parser.add_argument("--path", type=str, required=True, help="Path to the first .obj file")
    parser.add_argument("--num_centroids", type=int, default=2048, help="Number of centroids to sample")
    parser.add_argument("--num_samples", type=int, default=32, help="Number of samples per centroid")
    parser.add_argument("--radius", type=float, default=0.5, help="Radius for Grouping")
    parser.add_argument("--device", type=str, default='cpu', choices=['cpu', 'cuda'], help="Device to run the operations on (cpu or cuda)")
    
    return parser.parse_args()

# Load the .obj files
def load_mesh(file_path):
    return trimesh.load(file_path)

# Define a function to benchmark FPS and Grouping on CPU or GPU
def benchmark_fps_grouping(vertices, num_centroids, num_samples, radius, device):
    vertices = vertices.to(device)

    # Measure FPS time
    start_time = time.time()
    centroids_idx = FPS(vertices, num_centroids)
    centroids = index_point(vertices, centroids_idx)
    fps_time = time.time() - start_time

    # Measure Grouping time
    start_time = time.time()
    x_points, g_points, labels, idx = Grouping(vertices, vertices, centroids, num_samples, radius)
    grouping_time = time.time() - start_time

    return fps_time, grouping_time

if __name__ == "__main__":
    # Parse the command line arguments
    args = parse_args()

    # Load the .obj files
    mesh = load_mesh(args.path)
    vertices_tensor = torch.tensor(mesh.vertices, dtype=torch.float32).unsqueeze(0).to(args.device)

    # Print the benchmark info
    print(f"Number of points in the Original File: {vertices_tensor.shape[1]}")
    print(f"Number of Centroids: {args.num_centroids}")
    print(f"Number of Samples: {args.num_samples}")
    print(f"Total Number of points taken (#centroids \ times #samples): {args.num_centroids * args.num_samples}")
    print()

    # Run the benchmark on the specified device (CPU or GPU)
    fps_time, grouping_time = benchmark_fps_grouping(vertices_tensor, args.num_centroids, args.num_samples, args.radius, args.device)
    
    print(f"{args.device.upper()} - FPS Time: {fps_time:.4f}s, Grouping Time: {grouping_time:.4f}s, Total Time: {(fps_time + grouping_time):.4f}s")