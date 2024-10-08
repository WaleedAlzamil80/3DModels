import torch
from torch import nn
import numpy as np

from models.PointNetpp.PointNet2d import PointNet 
from sampling.PointsCloud.fps_grouping import fbsGrouping
import argparse

cuda = True if torch.cuda.is_available() else False
device = "cuda" if cuda else "cpu"

class SetApstractionLayer(nn.Module):
    '''
    inputs: 
            features: (Tensor) (batch_size, points, feature_maps)
            coordinates: (Tensor) (batch_size, points, coordinates)

    PointNet Input:
            features: (Tensor) (batch_size, feature_maps, num_centroids, num_samples)

    outputs:
            features: (Tensor) (batch_size, feature_maps, num_centroids, num_samples)
            coordinates: (Tensor) (batch_size, coordinates, num_centroids, num_samples)
            centroids: (Tensor) (batch_size, num_centroids, coordinates)
    '''

    def __init__(self, n_centroids, nsamples, radius, in_channels):
        super(SetApstractionLayer, self).__init__()
        self.n_centroids, self.nsamples, self.radius = n_centroids, nsamples, radius
        self.pointnet = PointNet(mode="features", input=in_channels)
        self.args = argparse.Namespace(**{'n_centroids': self.n_centroids, 'nsamples': self.nsamples, 'radius': self.radius})

    def forward(self, x, points):
        centroids, x_points, g_points, g_labels, idx = fbsGrouping(x, points, self.args)
        points, _, inT, feT = self.pointnet(g_points.permute(0, 3, 1, 2))

        return centroids, x_points.permute(0, 3, 1, 2), points, inT, feT

class PointNetCls(nn.Module):
    def __init__(self, mode="ddd", k=33):
        super(PointNetCls, self).__init__()
        self.as1 = SetApstractionLayer(16, 4, 0.5, 3)
        self.pointnet1 = PointNet(mode="features", input=1088)

        self.as2 = SetApstractionLayer(8, 2, 0.5, 1088)
        self.pointnet2 = PointNet(mode="features", input=1088)

        self.as3 = SetApstractionLayer(4, 2, 0.5, 1088)

        self.fc1 = nn.Linear(1088, 128)
        self.fc2 = nn.Linear(128, k)
        self.relu = nn.ReLU()

    def forward(self, x):
        points = x
        c, x, points, inT, feT = self.as1(x, points)
        points, _, _, _ = self.pointnet1(points)
        points = torch.max(points, dim=3)[0].transpose(1, 2)
        c, x, points, _, _ = self.as2(c, points)
        points, _, _, _ = self.pointnet1(points)
        points = torch.max(points, dim=3)[0].transpose(1, 2)
        c, x, points, _, _ = self.as3(c, points)

        x = torch.max(points, dim = 3)[0]
        x = torch.max(x, dim = 2)[0]

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x, inT, feT

class PointNetSeg(nn.Module):
    def __init__(self, mode="ddd", k=33):
        super(PointNetSeg, self).__init__()
        self.as1 = SetApstractionLayer(16, 4, 0.5, 3)
        # self.pointnet1 = PointNet(mode="features", input=1088)

        self.as2 = SetApstractionLayer(8, 2, 0.5, 1088)
        # self.pointnet2 = PointNet(mode="features", input=1088)

        self.as3 = SetApstractionLayer(4, 2, 0.5, 1088)

        self.fc1 = nn.Linear(1088, 128)
        self.fc2 = nn.Linear(128, k)
        self.relu = nn.ReLU()

    def forward(self, x):
        points = x

        c, x, points, inT, feT = self.as1(x, points)
        points = torch.max(points, dim=3)[0].transpose(1, 2)

        c, x, points, _, _ = self.as2(c, points)
        points = torch.max(points, dim=3)[0].transpose(1, 2)

        c, x, points, _, _ = self.as3(c, points)
        x = torch.max(points, dim = 3)[0]
        x = torch.max(x, dim = 2)[0]

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x, inT, feT

# Mode Factory that maps modes to classes
MODE_FACTORY = {
    "classification": PointNetCls,
    "segmentation": PointNetSeg,
}

def get_pointnetpp_mode(mode, *args, **kwargs):
    """Fetch the appropriate PointNet model based on the mode."""
    if mode not in MODE_FACTORY:
        raise ValueError(f"Mode {mode} is not available.")
    return MODE_FACTORY[mode](*args, **kwargs)

class PointNetpp(nn.Module):
    def __init__(self, mode="segmentation", k=33):
        super(PointNetpp, self).__init__()
        self.pointnetpp = get_pointnetpp_mode(mode=mode, k = k)

    def forward(self, x):
        return self.pointnetpp(x)