import torch
import torch.nn as nn
import numpy as np
import trimesh
from models.PointNet import PointNet
from losses.PointNetLosses import tnet_regularization
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set()

path = "/home/waleed/Documents/CROWNGENERATION/datasets/osfstorage-archive/all_data/training/lower/0EAKT1CU/0EAKT1CU_lower.obj"
mesh = trimesh.load(path)

# Convert to PyTorch tensors
vertices_tensor = torch.tensor(mesh.vertices, dtype=torch.float32).unsqueeze(0).transpose(1, 2)

print("Ready: ", vertices_tensor.shape)
# Check if faces exist and convert them too
faces_tensor = torch.tensor(mesh.faces, dtype=torch.long) if mesh.faces is not None else None

model = PointNet(mode="segmentation", k = 33)

x, inT, feT = model(vertices_tensor)
regurization = tnet_regularization(inT) + tnet_regularization(feT)
print("Total: ", regurization.item(),", input: ", tnet_regularization(inT).item(),", feature: ", tnet_regularization(feT).item())
print("Output: ", x.shape)
print(torch.argmax(x, dim=1))