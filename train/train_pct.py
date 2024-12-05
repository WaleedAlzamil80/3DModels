import os
import numpy as np
import torch
from factories.losses_factory import get_loss
from rigidTransformations import apply_random_transformation
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from utils.helpful import print_trainable_parameters
from metrics.meanAccClass import compute_mean_per_class_accuracy
from metrics.mIOU import compute_mIoU

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

    criterion = get_loss(args.loss)
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

            if args.rigid_augmentation_train:
                vertices = apply_random_transformation(vertices, rotat=args.rotat)

            vertices, labels, jaw = vertices.to(device), labels.to(device), jaw.to(device)

            # Forward pass
            outputs = model(vertices, jaw)

            loss = criterion(outputs, labels)
            cum_loss += loss.item()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Get predictions and true labels
            _, preds = torch.max(outputs, 2)

            # Append metric and loss to lists
            train_labels.extend(labels.view(-1).cpu().numpy())
            train_preds.extend(preds.view(-1).cpu().numpy())

            train_miou_e.append(compute_mIoU(preds.reshape(-1, args.n_centroids*args.nsamples), labels.reshape(-1, args.n_centroids*args.nsamples), args.k))
            train_acc_e.append(compute_mean_per_class_accuracy(preds.reshape(-1, args.n_centroids*args.nsamples), labels.reshape(-1, args.n_centroids*args.nsamples), args.k))

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
                vertices, labels, jaw = vertices.to(device), labels.to(device), jaw.to(device)

                if args.rigid_augmentation_test:
                    vertices = apply_random_transformation(vertices, rotat=args.rotat, trans=args.trans)

                # Forward pass
                outputs = model(vertices, jaw)

                t_loss += criterion(outputs, labels).item()

                # Get predictions and true labels
                _, preds = torch.max(outputs, 2)
                test_miou_e.append(compute_mIoU(preds.reshape(-1, args.n_centroids*args.nsamples), labels.reshape(-1, args.n_centroids*args.nsamples), args.k))
                test_acc_e.append(compute_mean_per_class_accuracy(preds.reshape(-1, args.n_centroids*args.nsamples), labels.reshape(-1, args.n_centroids*args.nsamples), args.k))

                test_labels.extend(labels.view(-1).cpu().numpy())
                test_preds.extend(preds.view(-1).cpu().numpy())

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

    return train_miou, test_miou, train_acc, test_acc, train_accuracy, test_accuracy, train_loss, test_loss
