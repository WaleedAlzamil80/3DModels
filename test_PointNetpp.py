import torch
import torch.nn as nn
import numpy as np
import trimesh
import trimesh.schemas
from models.PointNetPP import PointNetpp

from models.PointNetpp.FPS import FPS
from models.PointNetpp.Grouping import Grouping, index_point
from losses.PointNetLosses import tnet_regularization
from utils.helpful import print_trainable_parameters
from vis.visulizeGrouped import visualize_with_trimesh
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

path1 = "/home/waleed/Documents/3DLearning/3DModels/dataset/data_part_2/lower/HDVYC7UQ/HDVYC7UQ_lower.obj" # 121337
path2 = "/home/waleed/Documents/3DLearning/3DModels/dataset/data_part_4/upper/0EJBIPTC/0EJBIPTC_upper.obj" # 104444
mesh = trimesh.load(path1)
mesh2 = trimesh.load(path2)
mesh1 = mesh

# Calculate the centroid (which we'll use as the origin)
origin = mesh.centroid

# Define length for the axes
axis_length = 1.0

# Define the axis directions (unit vectors scaled by axis_length)
x_axis = np.array([[origin[0], origin[1], origin[2]], [origin[0] + axis_length, origin[1], origin[2]]])
y_axis = np.array([[origin[0], origin[1], origin[2]], [origin[0], origin[1] + axis_length, origin[2]]])
z_axis = np.array([[origin[0], origin[1], origin[2]], [origin[0], origin[1], origin[2] + axis_length]])

# Create trimesh Path3D objects for the axes
x_axis_path = trimesh.load_path(x_axis)
y_axis_path = trimesh.load_path(y_axis)
z_axis_path = trimesh.load_path(z_axis)

# Set colors for the axes
x_axis_path.colors = [[255, 0, 0, 255]]  # Red for X
y_axis_path.colors = [[0, 255, 0, 255]]  # Green for Y
z_axis_path.colors = [[0, 0, 255, 255]]  # Blue for Z

# Create a scene with the mesh and the axis paths
scene = trimesh.Scene([mesh, x_axis_path, y_axis_path, z_axis_path])

# Visualize the mesh with the axes
scene.show()

# mesh1.show()
# vertices_tensor = torch.tensor(mesh1.vertices, dtype=torch.float32).unsqueeze(0)

# Convert to PyTorch tensors
vertices_tensor = torch.tensor(mesh1.vertices, dtype=torch.float32).unsqueeze(0)
# vertices_tensor2 = torch.tensor(mesh2.vertices, dtype=torch.float32)[:4096*20, :].unsqueeze(0) # .to('cuda')

# vertices_tensor = torch.cat([vertices_tensor1, vertices_tensor2], dim = 0)
num_centroids = 2048
num_samples = 32

centroids_idx = FPS(vertices_tensor, num_centroids)

centroids = index_point(vertices_tensor, centroids_idx)

x_points, g_points, labels, idx = Grouping(vertices_tensor, vertices_tensor, centroids, num_samples, 0.5)

ss = x_points.reshape(-1, 3)
visualize_with_trimesh(ss, labels.squeeze(0)[:ss.shape[0]])
 
# Check if faces exist and convert them too
faces_tensor = torch.tensor(mesh1.faces, dtype=torch.long) if mesh1.faces is not None else None

model = PointNetpp(k=3)
print_trainable_parameters(model)
x = model(vertices_tensor)
print(x.shape)