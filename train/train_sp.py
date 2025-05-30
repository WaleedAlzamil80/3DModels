import os
import torch
import math
from factories.losses_factory import get_loss
from rigidTransformations import apply_random_transformation
from losses.RegularizarionPointNet import tnet_regularization
from tqdm import tqdm
from utils.helpful import print_trainable_parameters

cuda = True if torch.cuda.is_available() else False
device = 'cuda' if cuda else 'cpu'

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
            verticesTransformed = apply_random_transformation(vertices, rotat=args.rotat)
            vertices = vertices - vertices.mean(dim=1, keepdim=True)

            # Forward pass
            tin = model(verticesTransformed)
            verticesTransformed = torch.bmm(verticesTransformed, tin)
            # disvt = torch.cdist(verticesTransformed, verticesTransformed)
            # disv = torch.cdist(vertices, vertices)
            # pairwise_difference = torch.mean(torch.abs(disv-disvt), dim=(1,2))

            loss = criterion(verticesTransformed, vertices) + tnet_regularization(tin) # + pairwise_difference # + Orthogonality_Loss
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
                vertices = vertices - vertices.mean(dim=1, keepdim=True)

                # Forward pass
                tin = model(verticesTransformed)
                verticesTransformed = torch.bmm(verticesTransformed, tin)

                # disvt = torch.cdist(verticesTransformed, verticesTransformed)
                # disv = torch.cdist(vertices, vertices)
                # pairwise_difference = torch.mean(torch.abs(disv-disvt), dim=(1,2))

                t_loss += (criterion(verticesTransformed, vertices).item() + tnet_regularization(tin).item()) # + pairwise_difference.item())

        t_loss /= len(test_loader)

        print(f'Epoch [{epoch + 1}/{args.num_epochs}], train_Loss: {cum_loss:.4f}')
        print(f'Epoch [{epoch + 1}/{args.num_epochs}], test_Loss: {t_loss:.4f}')
        print("----------------------------------------------------------------------------------------------")
    print('Training finished.')

    torch.save(model.state_dict(), os.path.join(args.output, f"{args.model}_{epoch + 1}.pth"))

    return None, None, None, None, None, None, train_loss, test_loss
