import torch
import torch.nn as nn
import numpy as np
import trimesh
from models.PointNetPP import PointNetpp
from losses.PointNetLosses import tnet_regularization
from utils.helpful import print_trainable_parameters
from vis.visulizeGrouped import visualize_with_trimesh
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

path1 = "/home/waleed/Documents/3DLearning/3DModels/dataset/data_part_2/lower/HDVYC7UQ/HDVYC7UQ_lower.obj" # 121337
path2 = "/home/waleed/Documents/3DLearning/3DModels/dataset/data_part_4/upper/0EJBIPTC/0EJBIPTC_upper.obj" # 104444
mesh1 = trimesh.load(path1)
mesh2 = trimesh.load(path2)

# Convert to PyTorch tensors
vertices_tensor1 = torch.tensor(mesh1.vertices, dtype=torch.float32)[:4096*20, :].unsqueeze(0)
vertices_tensor2 = torch.tensor(mesh2.vertices, dtype=torch.float32)[:4096*20, :].unsqueeze(0) # .to('cuda')

vertices_tensor = torch.cat([vertices_tensor1, vertices_tensor2], dim = 0)
# centroids_idx = FPS(vertices_tensor, 4096 * 2)
# 
# centroids = index_point(vertices_tensor, centroids_idx)
# 
# x_points, g_points, labels, idx = Grouping(vertices_tensor, vertices_tensor, centroids, 8, 0.5)
# 
# ss = x_points.reshape(-1, 3)
# visualize_with_trimesh(ss, labels.squeeze(0)[:ss.shape[0]])
# 
# Check if faces exist and convert them too
faces_tensor = torch.tensor(mesh1.faces, dtype=torch.long) if mesh1.faces is not None else None

model = PointNetpp(k=3)
print_trainable_parameters(model)
x = model(vertices_tensor)
print(x.shape)