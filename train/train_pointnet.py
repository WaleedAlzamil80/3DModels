import os
import numpy as np
import torch
import torch.nn as nn
from losses.PointNetLosses import tnet_regularization
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from utils.helpful import print_trainable_parameters
from metrics.meanAccClass import compute_mean_per_class_accuracy
from metrics.mIOU import compute_mIoU
from vis.visulizeGrouped import visualize_with_trimesh

cuda = True if torch.cuda.is_available() else False
device = 'cuda' if cuda else 'cpu'

def train(model, train_loader, test_loader, args):

    train_accuracy = []
    train_loss = []
    test_accuracy = []
    test_loss = []

    train_miou = []
    test_miou = []
    train_acc = []
    test_acc = []

    print_trainable_parameters(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    for epoch in range(args.num_epochs):
        cum_loss = 0
        train_labels = []
        train_preds = []
        train_miou_e = []
        test_miou_e = []
        train_acc_e = []
        test_acc_e = []

        for vertices, labels, jaw in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}'):

            vertices, labels = vertices.to(device), labels.to(device).view(-1)

            # Forward pass
            outputs, tin, tfe = model(vertices)

            rtin, rtfe = tnet_regularization(tin), tnet_regularization(tfe)
            outputs = outputs.reshape(-1, args.k)
            loss = criterion(outputs, labels) + rtin + 0.001 * rtfe
            cum_loss += loss.item()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Get predictions and true labels
            _, preds = torch.max(outputs, 1)

            train_miou_e.append(compute_mIoU(preds.reshape(-1, args.n_centroids*args.nsamples), labels.reshape(-1, args.n_centroids*args.nsamples), args.k))
            train_acc_e.append(compute_mean_per_class_accuracy(preds.reshape(-1, args.n_centroids*args.nsamples), labels.reshape(-1, args.n_centroids*args.nsamples), args.k))

            # Append metric  and loss to lists
            train_labels.extend(labels.cpu().numpy())
            train_preds.extend(preds.cpu().numpy())

        # Calculate metrics
        train_epoch_accuracy = accuracy_score(train_labels, train_preds)
        train_miou.append(np.array(train_miou_e).mean())
        train_acc.append(np.array(train_acc_e).mean())

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
                vertices, labels = vertices.to(device), labels.to(device).view(-1)

                # Forward pass
                outputs, tin, tfe = model(vertices)

                outputs = outputs.view(-1, args.k)
                t_loss += criterion(outputs, labels).item() + tnet_regularization(tin).item() + 0.001 * tnet_regularization(tfe).item()

                # Get predictions and true labels
                _, preds = torch.max(outputs, 1)
                test_miou_e.append(compute_mIoU(preds.reshape(-1, args.n_centroids*args.nsamples), labels.reshape(-1, args.n_centroids*args.nsamples), args.k))
                test_acc_e.append(compute_mean_per_class_accuracy(preds.reshape(-1, args.n_centroids*args.nsamples), labels.reshape(-1, args.n_centroids*args.nsamples), args.k))

                test_labels.extend(labels.cpu().numpy())
                test_preds.extend(preds.cpu().numpy())

        # Calculate metrics
        test_epoch_accuracy = accuracy_score(test_labels, test_preds)
        test_miou.append(np.array(test_miou_e).mean())
        test_acc.append(np.array(test_acc_e).mean())

        # Calculate average loss
        t_loss /= len(test_loader)

        # Append metric  and loss to lists
        test_accuracy.append(test_epoch_accuracy)
        test_loss.append(t_loss)

        print(f'Epoch [{epoch + 1}/{args.num_epochs}], train_Loss: {cum_loss:.4f}, Accuracy: {train_epoch_accuracy:.4f}, mIOU: {train_miou[-1]:.4f}, Accuracy per Class: {train_acc[-1]:.4f}')
        print(f'Epoch [{epoch + 1}/{args.num_epochs}], test_Loss: {t_loss:.4f}, Accuracy: {test_epoch_accuracy:.4f},  mIOU: {test_miou[-1]:.4f}, Accuracy per Class: {test_acc[-1]:.4f}')
        print("----------------------------------------------------------------------------------------------")
    print('Training finished.')

    torch.save(model.state_dict(), os.path.join(args.output, f"{args.model}_{epoch + 1}.pth"))

    return train_accuracy, test_accuracy, train_loss, test_loss
