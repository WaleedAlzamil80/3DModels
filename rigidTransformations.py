import torch
import math

# Function to generate a random rotation matrix
def random_rotation_matrix(batch_size, rotat=0.25, device='cpu'):
    angles = (torch.rand(batch_size, 3, device=device) * 2 - 1) * math.pi * rotat
    cos, sin = torch.cos(angles), torch.sin(angles)

    # Rotation matrices for each axis
    R_x = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    R_y = R_x.clone()
    R_z = R_x.clone()

    R_x[:, 1, 1], R_x[:, 1, 2], R_x[:, 2, 1], R_x[:, 2, 2] = cos[:, 0], -sin[:, 0], sin[:, 0], cos[:, 0]
    R_y[:, 0, 0], R_y[:, 0, 2], R_y[:, 2, 0], R_y[:, 2, 2] = cos[:, 1], sin[:, 1], -sin[:, 1], cos[:, 1]
    R_z[:, 0, 0], R_z[:, 0, 1], R_z[:, 1, 0], R_z[:, 1, 1] = cos[:, 2], -sin[:, 2], sin[:, 2], cos[:, 2]

    # Combine rotations
    R = R_z @ R_y @ R_x
    return R

# Function to apply random rigid transformation
def apply_random_transformation(points, rotat = 0.25):
    batch_size, num_points, _ = points.shape
    device = points.device

    # Generate random rotations and translations
    R = random_rotation_matrix(batch_size, rotat = rotat, device=device)  # Shape: (batch_size, 3, 3)
    # t = (torch.rand(batch_size, 1, 3, device=device) * 2 - 1) * trans  # Shape: (batch_size, 1, 3), range [-trans, trans]

    # Apply the transformation
    transformed_points = torch.bmm(points, R.transpose(1, 2)) # + t  # (batch_size, num_points, 3)
    return transformed_points
