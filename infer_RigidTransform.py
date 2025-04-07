import os
import torch
import numpy as np
import fastmesh as fm
import trimesh

from factories.model_factory import get_model
from factories.sampling_factory import get_sampling_technique
from rigidTransformations import apply_random_transformation
from config.args_config import parse_args
from losses.RegularizarionPointNet import tnet_regularization
from factories.losses_factory import get_loss
args = parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
 
# Use the factory to dynamically get the model
model = get_model(args.model, mode=args.mode, k=args.k).to(device)

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
 
vertices_np = vertices_np - np.mean(vertices_np, axis=0)

sampling = get_sampling_technique(args.sampling)
vertices_np, idx = sampling(vertices_np, args.n_centroids, args.nsamples)
cloud = trimesh.points.PointCloud(vertices_np)
cloud.show()

vertices_ori = torch.tensor(vertices_np, dtype=torch.float32, device=device).view(-1, 3).unsqueeze(0)
vertices = torch.tensor(vertices_np, dtype=torch.float32, device=device).view(-1, 3).unsqueeze(0)

vertices = apply_random_transformation(vertices, rotat=args.rotat)
cloud = trimesh.points.PointCloud(vertices[0].cpu().detach())
cloud.show()

inT = model(vertices.transpose(1, 2).unsqueeze(2))
output = torch.matmul(vertices.squeeze(0), inT.squeeze(0)).unsqueeze(0)

# How far the matrix from identity
R = inT
det_R = torch.det(R[0])
is_orthogonal = torch.allclose(R[0] @ R[0].T, torch.eye(3).to(device), atol=1e-5)

loss1 = get_loss("chamfer")
loss2 = get_loss("l2")
print(f"Chamfer Loss: {loss1(output, vertices_ori)}")
print(f"L2 Loss: {loss2(output, vertices_ori)}")

print(f"Regularization to prevent the model from shearing or scaling: {tnet_regularization(inT)}")
print(f"Determinant: {det_R}, Is Orthogonal: {is_orthogonal}")

# Create a trimesh object for the point cloud
cloud = trimesh.points.PointCloud(output[0].cpu().detach())
# Show the point cloud
cloud.show()

# Pairwise comparisons for all combinations of tensors
tol = 1e-3
distances_vertices_ori = torch.cdist(vertices_ori, vertices_ori)
distances_vertices = torch.cdist(vertices, vertices)
distances_output = torch.cdist(output, output)

# MY newwwww Loss Function
print((torch.mean(torch.abs(distances_vertices_ori-distances_output), dim=(1,2))).item())
print((torch.mean(torch.abs(distances_vertices-distances_output), dim=(1,2))).item())
print((torch.mean(torch.abs(distances_vertices-distances_vertices_ori), dim=(1,2))).item())
