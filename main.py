import os
import torch
import torch.nn as nn

from factories.dataset_factory import get_dataset_loader
from factories.model_factory import get_model
from train.train import get_train

from vis.plots import plot_training_data
from config.args_config import parse_args

args = parse_args()

cuda = True if torch.cuda.is_available() else False
device = 'cuda' if cuda else 'cpu'

# Use the factory to dynamically get the dataloaders for specific dataset
train_loader, test_loader = get_dataset_loader(args.Dataset, args)

# Use the factory to dynamically get the model
model = get_model(args.model, mode=args.mode, k=args.k).to(device)

# Wrap the model in DataParallel for multi-GPU training
model = nn.DataParallel(model).to(device)

if not os.path.exists(args.output):
    os.makedirs(args.output)

# Train the model
train_miou, test_miou, train_acc, test_acc, train_accuracy, test_accuracy, train_loss, test_loss = get_train(args.model, model, train_loader, test_loader, args)

# Save the plots
plot_training_data(train_miou, test_miou, train_acc, test_acc, train_accuracy, test_accuracy, train_loss, test_loss, args.output)