import os

import torch
from torch import nn
import numpy as np
import fastmesh as fm
import trimesh
import json

from factories.model_factory import get_model
from factories.sampling_factory import get_sampling_technique
from config.args_config import parse_args

args = parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
 
# Use the factory to dynamically get the model
model = get_model(args.model, mode=args.mode, k=args.k).to(device)
model = nn.DataParallel(model).to(device)

# Load pretrained weights if provided
if args.pretrained and os.path.exists(args.pretrained):
    try:
        print(f"Loading pretrained model from {args.pretrained}")
        state_dict = torch.load(args.pretrained, map_location=device)

        # Strip "module." prefix if the model was saved with DataParallel
        state_dict = {key[7:] if key.startswith("module.") else key: value for key, value in state_dict.items()}
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Failed to load pretrained model from {args.pretrained}. Error: {e}")
        print(f"Ensure the checkpoint matches the architecture '{args.model}'.")
else:
    print(f"No pretrained weights found. Initializing a new '{args.model}' model.")


model.eval()

vertices_np = fm.load(args.path)[0] # np.array(mesh.vertices)
if args.clean:
    origin = np.mean(vertices_np, axis=0)

    z_values = vertices_np[:, 2]
    y_values = vertices_np[:, 1]
    x_values = vertices_np[:, 0]

    y_mean = np.mean(y_values)
    y_std = np.std(y_values)
    x_mean = np.mean(x_values)
    x_std = np.std(x_values)
    alpha = 2.0

    valid_mask = (z_values > (origin[2] - 5)) & \
                    (y_values < (y_mean + alpha * y_std)) & (y_values > (y_mean - alpha * y_std)) & \
                    (x_values < (x_mean + alpha * x_std)) & (x_values > (x_mean - alpha * x_std))

    vertices_np = vertices_np[valid_mask]

# vertices_np = vertices_np - np.mean(vertices_np, axis=0)

sampling = get_sampling_technique(args.sampling)
vertices_np, idx = sampling(vertices_np, args.n_centroids, args.nsamples)
vertices = torch.tensor(vertices_np, dtype=torch.float32, device=device).view(-1, 3).unsqueeze(0)

inT = model(vertices.transpose(1, 2).unsqueeze(2))
output = torch.matmul(vertices.squeeze(0), inT.squeeze(0))

# Create a trimesh object for the point cloud
cloud = trimesh.points.PointCloud(output.cpu().detach())
# Show the point cloud
cloud.show()