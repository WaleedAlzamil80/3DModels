import torch
import torch.nn as nn
import argparse

from Dataset.segmentation_OSF.Dataset import get_data_loaders 
from models.PointNet import PointNet
from train import train
def parse_args():
    parser = argparse.ArgumentParser(description="Model training parameters")
    
    parser.add_argument('--num_epochsbatch_size', type=int, default=10, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of Workers")
    parser.add_argument('--path', type=str, default="dataset", help="Path of the dataset")

    return parser.parse_args()

args = parse_args()

cuda = True if torch.cuda.is_available() else False
device = 'cuda' if cuda else 'cpu'

train_loader, test_loader = get_data_loaders(args.path, args.batch_size)
model = PointNet(mode = "segmentation", k = 33).to(device)
model = nn.DataParallel(model).to(device)

train(model, train_loader, test_loader, args)