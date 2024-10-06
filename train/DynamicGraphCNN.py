import os
import torch
import torch.nn as nn
from losses.PointNetLosses import tnet_regularization
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from utils.helpful import print_trainable_parameters

cuda = True if torch.cuda.is_available() else False
device = 'cuda' if cuda else 'cpu'

def train(model, train_loader, test_loader, args):

    train_accuracy = []
    train_loss = []
    test_accuracy = []
    test_loss = []
    print(model)
    print_trainable_parameters(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    for epoch in range(args.num_epochs):
        cum_loss = 0
        train_labels = []
        train_preds = []

        for vertices, labels, jaw in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}'):

            vertices, labels, jaw = vertices.to(device), labels.to(device), jaw.to(device)

            # Forward pass
            outputs, tin = model(vertices, jaw)
            print(outputs.shape)
            print(labels.shape)
            print(jaw.shape)
            rtin = tnet_regularization(tin)
            loss = criterion(outputs.reshape(-1, args.k), labels.reshape(-1)) + rtin
            cum_loss += loss.item() + rtin

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Get predictions and true labels
            _, preds = torch.max(outputs, 1)

            # Append metric  and loss to lists
            train_labels.extend(labels.cpu().numpy())
            train_preds.extend(preds.cpu().numpy())

        # Calculate metrics
        train_epoch_accuracy = accuracy_score(train_labels, train_preds)

        # Calculate average loss
        cum_loss /= len(train_loader)

        train_accuracy.append(train_epoch_accuracy)
        train_loss.append(cum_loss)


        model.eval()
        test_labels = []
        test_preds = []
        t_loss = 0

        with torch.no_grad():
            for vertices, labels, jaw in tqdm(test_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}'):
                vertices, labels, jaw = vertices.to(device), labels.to(device).view(-1), jaw.to(device)

                # Forward pass
                outputs, tin = model(vertices, jaw)
                outputs = outputs.view(-1, args.k)
                t_loss += criterion(outputs, labels).item() + tnet_regularization(tin).item()

                # Get predictions and true labels
                _, preds = torch.max(outputs, 1)

                test_labels.extend(labels.cpu().numpy())
                test_preds.extend(preds.cpu().numpy())

        # Calculate metrics
        test_epoch_accuracy = accuracy_score(test_labels, test_preds)

        # Calculate average loss
        t_loss /= len(test_loader)

        # Append metric  and loss to lists
        test_accuracy.append(test_epoch_accuracy)
        test_loss.append(t_loss)


        print(f'Epoch [{epoch + 1}/{args.num_epochs}], train_Loss: {cum_loss:.4f}, Accuracy: {train_epoch_accuracy:.4f}')
        print(f'Epoch [{epoch + 1}/{args.num_epochs}], test_Loss: {t_loss:.4f}, Accuracy: {test_epoch_accuracy:.4f}')
        print("----------------------------------------------------------------------------------------------")
    print('Training finished.')

    torch.save(model.state_dict(), os.path.join(args.output, f"model_epoch_{epoch}.pth"))

    return train_accuracy, test_accuracy, train_loss, test_loss