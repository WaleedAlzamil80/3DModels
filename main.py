import torch
import torch.nn as nn
import argparse

from Dataset.segmentation_OSF.Dataset import get_data_loaders 
from Dataset.modelnet.Dataloader import ModelNet10Dataset
from models.PointNet import PointNet
from models.PointNetPP import PointNetpp
from train import train
from vis.plots import plot_training_data

def parse_args():
    parser = argparse.ArgumentParser(description="Model training parameters")

    parser.add_argument('--num_epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of Workers")
    parser.add_argument('--path', type=str, default="dataset", help="Path of the dataset")
    parser.add_argument('--Dataset', type=str, default="OSF", help="Which Dataset?")

    parser.add_argument('--output', type=str, default="output", help="Output path")

    parser.add_argument('--test_ids', type=str, default="private-testing-set.txt", help="Path of the ids dataset for testing")
    parser.add_argument('--p', type=int, default=3, help="data parts")

    parser.add_argument('--k', type=int, default=33, help="Number classes")
    parser.add_argument('--model', type=str, default="segmentation", help="Select the model")
    parser.add_argument('--mode', type=str, default="segmentation", help="Problems ex:- segmentaion, classification")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning Rate")

    return parser.parse_args()

args = parse_args()

cuda = True if torch.cuda.is_available() else False
device = 'cuda' if cuda else 'cpu'

train_loader, test_loader = get_data_loaders(args)

if args.model == "PointNet":
    model = PointNet(mode = args.mode, k = args.k).to(device)
elif args.model == "PointNet++":
    model = PointNetpp(mode = args.mode, k = args.k).to(device)

model = nn.DataParallel(model).to(device)

train_accuracy, test_accuracy, train_loss, test_loss = train(model, train_loader, test_loader, args)

# Save the plots
plot_training_data(train_accuracy, test_accuracy, train_loss, test_loss, args.output)