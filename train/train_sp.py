import os
import torch
import math
from factories.losses_factory import get_loss
from tqdm import tqdm
from utils.helpful import print_trainable_parameters

cuda = True if torch.cuda.is_available() else False
device = 'cuda' if cuda else 'cpu'

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
def apply_random_transformation(points, rotat = 0.25, trans = 0.5):
    batch_size, num_points, _ = points.shape
    device = points.device

    # Generate random rotations and translations
    R = random_rotation_matrix(batch_size, totat = rotat, device=device)  # Shape: (batch_size, 3, 3)
    t = (torch.rand(batch_size, 1, 3, device=device) * 2 - 1) * trans  # Shape: (batch_size, 1, 3), range [-trans, trans]

    # Apply the transformation
    transformed_points = torch.bmm(points, R.transpose(1, 2)) + t  # (batch_size, num_points, 3)
    return transformed_points

def train(model, train_loader, test_loader, args):

    train_loss = []
    test_loss = []

    print_trainable_parameters(model)

    criterion = get_loss(args.loss)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [5*(i+1) for i in range(50)], gamma = args.gamma)

    for epoch in range(args.num_epochs):
        cum_loss = 0

        for vertices, labels, jaw in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}'):

            vertices = vertices.to(device)
            verticesTransformed = apply_random_transformation(vertices, rotat=args.rotat, trans=args.trans)

            # Forward pass
            tin = model(verticesTransformed.transpose(1, 2).unsqueeze(3))
            verticesTransformed = torch.bmm(verticesTransformed, tin)

            loss = criterion(verticesTransformed, vertices)
            cum_loss += loss.item()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Calculate average loss
        cum_loss /= len(train_loader)

        train_loss.append(cum_loss)

        model.eval()
        t_loss = 0

        with torch.no_grad():
            for vertices, labels, jaw in tqdm(test_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}'):
                vertices = vertices.to(device)
                verticesTransformed = apply_random_transformation(vertices)

                # Forward pass
                tin = model(verticesTransformed.transpose(1, 2).unsqueeze(3))
                verticesTransformed = torch.bmm(verticesTransformed, tin)

                t_loss += criterion(verticesTransformed, vertices).item()

        t_loss /= len(test_loader)

        print(f'Epoch [{epoch + 1}/{args.num_epochs}], train_Loss: {cum_loss:.4f}')
        print(f'Epoch [{epoch + 1}/{args.num_epochs}], test_Loss: {t_loss:.4f}')
        print("----------------------------------------------------------------------------------------------")
    print('Training finished.')

    torch.save(model.state_dict(), os.path.join(args.output, f"{args.model}_{epoch + 1}.pth"))

    return None, None, None, None, None, None, train_loss, test_loss
