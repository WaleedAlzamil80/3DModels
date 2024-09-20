import torch
from torch import nn
import numpy as np

# the input shape should be (Batch_Size, In_channels, Centroids, Samples)

class TNetkd(nn.Module):
    def __init__(self, k = 3, mlp= [64, 128, 1024, 512, 256]):
        super(TNetkd, self).__init__()
        self.k=k
        self.conv1 = nn.Conv2d(in_channels=self.k, out_channels=mlp[0], kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=mlp[0], out_channels=mlp[1], kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=mlp[1], out_channels=mlp[2], kernel_size=1)

        self.conv4 = nn.Conv2d(in_channels=mlp[2], out_channels=mlp[3], kernel_size=1)
        self.conv5 = nn.Conv2d(in_channels=mlp[3], out_channels=mlp[4], kernel_size=1)
        self.conv6 = nn.Conv2d(in_channels=mlp[4], out_channels=self.k*self.k, kernel_size=1)

        self.bn1 = nn.BatchNorm2d(num_features=mlp[0])
        self.bn2 = nn.BatchNorm2d(num_features=mlp[1])
        self.bn3 = nn.BatchNorm2d(num_features=mlp[2])
        self.bn4 = nn.BatchNorm2d(num_features=mlp[3])
        self.bn5 = nn.BatchNorm2d(num_features=mlp[4])
        self.relu = nn.ReLU()

    def forward(self, x):
        bs, _, c, n = x.shape
        x = self.relu(self.bn1(self.conv1(x)))                                          
        x = self.relu(self.bn2(self.conv2(x)))                                          
        x = self.relu(self.bn3(self.conv3(x)))                                                
        x = self.relu(self.bn4(self.conv4(x)))                                                
        x = self.relu(self.bn5(self.conv5(x)))                                                
        x = self.conv6(x)                                                                     

        x = torch.max(x, dim=3, keepdim=False)[0]                                            # (1, k * k, C)
        x = torch.max(x, dim=2, keepdim=False)[0]                                            # (1, k * k, C)

        iden = torch.eye(self.k, requires_grad=True).view(1, self.k * self.k).expand(bs, -1).to(x.device)
        x = x + iden
        x = x.view(bs, self.k, self.k)                                                   # (1, k, k)
        return x

class PointNetGfeat(nn.Module):
    def __init__(self, k = 3, mlp = [64, 64, 64, 128, 1024], global_features = True):
        super(PointNetGfeat, self).__init__()
        self.k = k
        self.global_features = global_features
        self.Tnet3d = TNetkd(self.k)
        self.conv1 = nn.Conv1d(in_channels=self.k, out_channels=64, kernel_size=1)
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
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.logsoftmax(self.conv4(x))
        return x, inTra, feTra

class PointNet(nn.Module):
    def __init__(self, mode = "classification", k=28, input=3):
        super(PointNet, self).__init__()
        self.k = k
        self.input = input
        if mode == "classification":
            self.PointNet = PointNetCls(k, self.input)
        elif mode == "segmentation":
            self.PointNet = PointNetSeg(k, self.input)
        elif mode == "features":
            self.PointNet = PointNetGfeat(k, self.input)

    def forward(self, x):
        return self.PointNet(x.transpose(1, 2))

class PointNetPartSeg(nn.Module):
    def __init__(self, k=33, input = 3):
        super(PointNetPartSeg, self).__init__()

    def forward(self, x):
        return x