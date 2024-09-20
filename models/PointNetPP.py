import torch
from torch import nn
import numpy as np

from models.PointNetpp.FPS import FPS
from models.PointNetpp.Grouping import Grouping, index_point
from vis.visulizeGrouped import visualize_with_trimesh

cuda = True if torch.cuda.is_available() else False
device = "cuda" if cuda else "cpu"

class SetApstractionLayer(nn.Module):
    def __init__(self, n_centroids, nsamples, radius, in_channels, mlp):
        super(SetApstractionLayer, self).__init__()
        self.n_centroids, self.nsamples, self.radius = n_centroids, nsamples, radius

        self.conv1 = nn.Conv2d(in_channels, mlp[0], kernel_size=1)
        self.conv2 = nn.Conv2d(mlp[0], mlp[1], kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mlp[0])
        self.bn2 = nn.BatchNorm2d(mlp[1])
        self.relu = nn.ReLU()

    def forward(self, x, points):
        centroids_idx = FPS(x, self.n_centroids)
        centroids = index_point(x, centroids_idx)
        x_points, g_points, labels, idx = Grouping(x, points, centroids, self.nsamples, self.radius)

        points = self.relu(self.bn1(self.conv1(g_points.transpose(1, 3))))
        points = self.relu(self.bn2(self.conv2(points)))
        points = torch.max(points, dim = 2)[0].transpose(1, 2)

        return centroids, points

class PointNetpp(nn.Module):
    def __init__(self, mode="segmentation", k=33):
        super(PointNetpp, self).__init__()
        self.as1 = SetApstractionLayer(4096, 32, 0.5, 3, mlp=[64, 128])
        self.as2 = SetApstractionLayer(1024, 64, 0.5, 128, mlp=[128, 256])
        self.as3 = SetApstractionLayer(256, 128, 0.5, 256, mlp=[256, 512])

        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, k)
        self.relu = nn.ReLU()

    def forward(self, x):
        points = x
        x, points = self.as1(x, points)
        x, points = self.as2(x, points)
        x, points = self.as3(x, points) # output (Batch_size, centroids, D)

        x = torch.max(points, dim = 1)[0]

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x