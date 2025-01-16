import torch
from torch import nn
import numpy as np

# the input shape should be (Batch_Size, In_channels, Sequence_Length)
class TNetkd(nn.Module):
    def __init__(self, k = 3):
        super(TNetkd, self).__init__()
        self.k=k
        self.conv1 = nn.Conv1d(in_channels=self.k, out_channels=64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1, bias=False)

        self.conv4 = nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1, bias=False)
        self.conv6 = nn.Conv1d(in_channels=256, out_channels=self.k*self.k, kernel_size=1)

        # self.fc1 = nn.Linear(in_features=1024, out_features=512)
        # self.fc2 = nn.Linear(in_features=512, out_features=256)
        # self.fc3 = nn.Linear(in_features=256, out_features=self.k*self.k)

        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.bn2 = nn.BatchNorm1d(num_features=128)
        self.bn3 = nn.BatchNorm1d(num_features=1024)
        self.bn4 = nn.BatchNorm1d(num_features=512)
        self.bn5 = nn.BatchNorm1d(num_features=256)
        self.relu = nn.ReLU()

    def forward(self, x):                                                                   # x.shape = # (1, k, #pointClouds)
        bs, _, n = x.shape
        x = self.relu(self.bn1(self.conv1(x)))                                          
        x = self.relu(self.bn2(self.conv2(x)))                                          
        x = self.relu(self.bn3(self.conv3(x)))                                                # (1, 1024, #pointClouds)
        # both do exactly the same thing
        # x = self.relu(self.bn4(self.fc1(x.transpose(1, 2)).transpose(1, 2)))                # (1, #pointClouds, 512)
        # x = self.relu(self.bn5(self.fc2(x.transpose(1, 2)).transpose(1, 2)))                # (1, #pointClouds, 256)
        # x = self.fc3(x.transpose(1, 2))                                                     # (1, #pointClouds, k * k)  max with dim = 1
        x = self.relu(self.bn4(self.conv4(x)))                                                # (1, 512, #pointClouds)
        x = self.relu(self.bn5(self.conv5(x)))                                                # (1, 256, #pointClouds)
        x = self.conv6(x)                                                                     # (1, k * k,#pointClouds)   max with dim = 2

        x = torch.max(x, dim=2, keepdim=False)[0]                                            # (1, k * k)
        iden = torch.eye(self.k, requires_grad=True).view(1, self.k * self.k).expand(bs, -1).to(x.device)
        x = x + iden
        x = x.view(bs, self.k, self.k)                                                   # (1, k, k)
        return x

class PointNetGfeat(nn.Module):
    def __init__(self, k = 3, global_features = True):
        super(PointNetGfeat, self).__init__()
        self.k = k
        self.global_features = global_features
        self.Tnet3d = TNetkd(self.k)
        self.conv1 = nn.Conv1d(in_channels=self.k, out_channels=64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        self.Tnet64d = TNetkd(64)
        self.conv3 = nn.Conv1d(64, 64, 1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, 1, bias=False)
        self.conv5 = nn.Conv1d(128, 1024, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        self.relu = nn.ReLU()

    def forward(self, x):                                                             # x.shape = # (1, k, #pointClouds)
        bs, _, n = x.shape
        inT = self.Tnet3d(x)
        x = torch.bmm(x.transpose(1, 2), inT).transpose(1, 2)                          # (bs, n, 3) OP(BatchMatMul) (bs, 3, 3)  ----->    (N, n, 3)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        feT = self.Tnet64d(x)
        local_features = torch.bmm(x.view(bs, n, 64), feT).transpose(1, 2)            # (bs, 64, 1) 
        x = self.relu(self.bn3(self.conv3(local_features)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))                                        # (bs, 1024, 1)

        global_features = torch.max(x, dim = 2, keepdim=True)[0]                      # (1, 1024)

        if self.global_features:
            return global_features, inT, feT

        x = torch.cat([local_features, global_features.expand(-1, -1, n)], dim = 1)
        return x, inT, feT

class PointNetCls(nn.Module):
    def __init__(self, k = 28, input=3):
        super(PointNetCls, self).__init__()
        self.k = k
        self.input = input
        self.feNet = PointNetGfeat(k = self.input)
        self.fc1 = nn.Linear(in_features=1024, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=self.k)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop25 = nn.Dropout(0.25)
        self.drop70 = nn.Dropout(0.70)


        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        x, inTra, feTra = self.feNet(x)
        x = x.squeeze(2)
        x = self.relu(self.drop25(self.fc1(x)))
        x = self.relu(self.drop70(self.fc2(x)))        
        x = self.logsoftmax(self.fc3(x))
        
        return x, inTra, feTra

class PointNetSeg(nn.Module):
    def __init__(self, k = 28, input = 3):
        super(PointNetSeg, self).__init__()
        self.input = input
        self.k = k
        self.feNet = PointNetGfeat(k = self.input, global_features=False)
        self.conv1 = nn.Conv1d(1088, 512, 1, bias=False)
        self.conv2 = nn.Conv1d(512, 256, 1, bias=False)
        self.conv3 = nn.Conv1d(256, 128, 1, bias=False)
        self.conv4 = nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=-1)


    def forward(self, x):
        x, inTra, feTra = self.feNet(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.logsoftmax(self.conv4(x)).transpose(1, 2)
        return x, inTra, feTra

# Mode Factory that maps modes to classes
MODE_FACTORY = {
    "classification": PointNetCls,
    "segmentation": PointNetSeg,
    "features": PointNetGfeat,
}

def get_pointnet_mode(mode, *args, **kwargs):
    """Fetch the appropriate PointNet model based on the mode."""
    if mode not in MODE_FACTORY:
        raise ValueError(f"Mode {mode} is not available.")
    return MODE_FACTORY[mode](*args, **kwargs)


class PointNet(nn.Module):
    def __init__(self, mode = "classification", k=28, input=3):
        super(PointNet, self).__init__()
        self.PointNet = get_pointnet_mode(mode, k, input)

    def forward(self, x):
        # If input is (Batch_size, #channels, #points), we need to reshape it to (Batch_size, #channels, 1, #points)
        if len(x.shape) == 3:
            bs, channels, points = x.shape
            # x = x.unsqueeze(2)  # Add a new dimension to match Conv2d input format: (Batch_size, #channels, 1, #points)
        elif len(x.shape) == 4:
            bs, channels, centroids, points = x.shape
            # No need to change if it's already 4D

        return self.PointNet(x.transpose(1, 2))

class PointNetPartSeg(nn.Module):
    def __init__(self, k=33, input = 3):
        super(PointNetPartSeg, self).__init__()

    def forward(self, x):
        return x