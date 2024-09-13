import torch
from torch import nn
import numpy as np

class TNetkd(nn.Module):
    def __init__(self, k = 3):
        super(TNetkd, self).__init__()
        self.k=k
        self.conv1 = nn.Conv1d(in_channels=self.k, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1)
        self.fc1 = nn.Linear(in_features=1024, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=self.k*self.k)
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.bn2 = nn.BatchNorm1d(num_features=128)
        self.bn3 = nn.BatchNorm1d(num_features=1024)
        self.bn4 = nn.BatchNorm1d(num_features=512)
        self.bn5 = nn.BatchNorm1d(num_features=256)
        self.relu = nn.ReLU()

    def forward(self, x):                                                                   # x.shape = # (#pointClouds, k, 1)
        bs = x.shape[0]                                         
        x = self.relu(self.bn1(self.conv1(x)))                                          
        x = self.relu(self.bn2(self.conv2(x)))                                          
        x = self.relu(self.bn3(self.conv3(x)))                                              # (#pointClouds, 1024, 1)
        x = x.reshape(bs, 1024)                                                             # (#pointClouds, 1024)
        x = self.relu(self.bn4(self.fc1(x)))                                                # (#pointClouds, 512)
        x = self.relu(self.bn5(self.fc2(x)))                                                # (#pointClouds, 256)
        x = self.fc3(x)                                                                     # (#pointClouds, k * k)
        iden = torch.eye(self.k, requires_grad=True).view(1, self.k * self.k).repeat(bs, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.reshape(bs, self.k, self.k)                                                   # (#pointClouds, k, k)
        return x

class PointNetGfeat(nn.Module):
    def __init__(self, global_features = True):
        super(PointNetGfeat, self).__init__()
        self.global_features = global_features
        self.Tnet3d = TNetkd(3)
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        self.Tnet64d = TNetkd(64)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        self.relu = nn.ReLU()

    def forward(self, x):                                                             # x.shape = # (#pointClouds, k)
        bs = x.shape[0]                                         
        x = x.reshape(bs, 3, 1)                                                       # (#pointClouds, k, 1)
        inT = self.Tnet3d(x)
        x = torch.bmm(x.view(x.shape[0], 1, 3), inT).reshape(-1, 3, 1)                # (N, 1, 3) OP(BatchMatMul) (N, 3, 3)  ----->    (N, 1, 3)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        feT = self.Tnet64d(x)
        local_features = torch.bmm(x.view(x.shape[0], 1, 64), feT).reshape(-1, 64, 1) # (N, 64, 1) 
        x = self.relu(self.bn3(self.conv3(local_features)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))                                        # (N, 1024, 1)
        global_features, _ = torch.max(x, dim = 0)                                    # (1024, 1)
        global_features = global_features.reshape(1, 1024)

        if self.global_features:
            return global_features, inT, feT

        x = torch.cat([local_features.reshape(-1, 64), global_features.repeat(local_features.shape[0], 1)], dim = 1)
        return x, inT, feT

class PointNetCls(nn.Module):
    def __init__(self, k = 28):
        super(PointNetCls, self).__init__()
        self.k = k
        self.feNet = PointNetGfeat()
        self.fc1 = nn.Linear(in_features=1024, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=self.k)

        # you can't use BN as you only have a one value for each channel after aggregation
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(0.25)

        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        x, inTra, feTra = self.feNet(x)
        x = self.relu(self.drop(self.fc1(x)))
        x = self.relu(self.drop(self.fc2(x)))
        x = self.logsoftmax(self.fc3(x))
        return x, inTra, feTra

class PointNetSeg(nn.Module):
    def __init__(self, k = 28):
        super(PointNetSeg, self).__init__()
        self.k = k
        self.feNet = PointNetGfeat(global_features=False)
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=-1)


    def forward(self, x):
        x, inTra, feTra = self.feNet(x)
        x = self.relu(self.bn1(self.conv1(x.reshape(-1, 1088, 1))))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.logsoftmax(self.conv4(x))
        return x, inTra, feTra

class PointNet(nn.Module):
    def __init__(self, mode = "classification", k=28):
        super(PointNet, self).__init__()
        self.k = k
        if mode == "classification":
            self.PointNet = PointNetCls(k)
        elif mode == "segmentation":
            self.PointNet = PointNetSeg(k)
        elif mode == "features":
            self.PointNet = PointNetGfeat(k)

    def forward(self, x):
        return self.PointNet(x)