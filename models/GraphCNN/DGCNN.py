import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..PointNetpp.PointNet2d import TNetkd
from sampling.PointsCloud.knn import kdneighGPU

import argparse

class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=32):
        super(EdgeConv, self).__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels*2, out_channels=out_channels[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels[0]),
            nn.LeakyReLU(),
        )
        for i in range(1, len(out_channels)):
            self.conv.append(nn.Sequential(
            nn.Conv2d(in_channels=out_channels[i-1], out_channels=out_channels[i], kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels[i]),
            nn.LeakyReLU(),
        ))

        self.args = argparse.Namespace(**{'knn': self.k})

    def forward(self, x):
        x, y = kdneighGPU(x, self.args)

        x = self.conv(x.permute(0, 3, 1, 2))
        x = torch.max(x, dim=-1, keepdim=False)[0]
        return x.permute(0, 2, 1)

class DGCNNCls(nn.Module):
    def __init__(self, k):
        super(DGCNNCls, self).__init__()
        self.k = k

        self.Tnet3d = TNetkd(3)
        self.edgeconv1 = EdgeConv(3, [64])
        self.edgeconv2 = EdgeConv(64, [64])
        self.edgeconv3 = EdgeConv(64, [64])
        self.edgeconv4 = EdgeConv(64, [128])
        self.conv1 = nn.Conv1d(320, 1024, 1)
        self.conv2 = nn.Conv1d(1024, 512, 1)
        self.conv3 = nn.Conv1d(512, 128, 1)
        self.conv4 = nn.Conv1d(128, k, 1)

        self.relu = nn.LeakyReLU()

    def forward(self, x, jaw=0):

        inT = self.Tnet3d(x.transpose(1, 2).unsqueeze(3))
        x = torch.bmm(x, inT)

        x1 = self.edgeconv1(x)
        x2 = self.edgeconv2(x1)
        x3 = self.edgeconv3(x2)        
        x4 = self.edgeconv4(x3)

        x = torch.cat([x1, x2, x3, x4], dim=2).permute(0, 2, 1)
        x = torch.max(x, dim = 2, keepdim=True)[0]

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x).squeeze(2)

        return x, inT

class DGCNNSeg(nn.Module):
    def __init__(self, k):
        super(DGCNNSeg, self).__init__()
        self.k = k

        self.Tnet3d = TNetkd(3)

        self.embedding = nn.Embedding(2, 64)
        self.conv_emb = nn.Conv1d(64, 64, 1)

        self.edgeconv1 = EdgeConv(3, [64, 64])
        self.edgeconv2 = EdgeConv(64, [64, 64])
        self.edgeconv3 = EdgeConv(64, [64])

        self.conv1 = nn.Conv1d(192, 1024, 1)

        self.conv2 = nn.Conv1d(1280, 256, 1)
        self.conv3 = nn.Conv1d(256, 256, 1)
        self.conv4 = nn.Conv1d(256, 128, 1)
        self.conv5 = nn.Conv1d(128, k, 1)


        self.relu = nn.LeakyReLU()

    def forward(self, x, jaw=0):

        inT = self.Tnet3d(x.transpose(1, 2).unsqueeze(3))
        x = torch.bmm(x, inT)

        emb = self.embedding(jaw).unsqueeze(2)
        emb = self.conv_emb(emb).permute(0, 2, 1)

        x1 = self.edgeconv1(x)
        x2 = self.edgeconv2(x1)
        x3 = self.edgeconv3(x2)        

        x = torch.cat([x1, x2, x3], dim=2).permute(0, 2, 1)
        x = torch.max(x, dim = 2, keepdim=True)[0]
        x = self.relu(self.conv1(x)).permute(0, 2, 1)
        x = torch.cat([x, emb], dim=2)

        x = torch.cat([x.expand(x1.size(0), x1.size(1), -1), x1, x2, x3], dim=2).permute(0, 2, 1)

        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x).transpose(1, 2)

        return x, inT

# Mode Factory that maps modes to classes
MODE_FACTORY = {
    "classification": DGCNNCls,
    "segmentation": DGCNNSeg,
}

def get_dgcnn_mode(mode, *args, **kwargs):
    """Fetch the appropriate PointNet model based on the mode."""
    if mode not in MODE_FACTORY:
        raise ValueError(f"Mode {mode} is not available.")
    return MODE_FACTORY[mode](*args, **kwargs)


class DGCNN(nn.Module):
    def __init__(self, mode, k):
        super(DGCNN, self).__init__()
        self.dgcnn = get_dgcnn_mode(mode, k)

    def forward(self, x, jaw=0):
        return self.dgcnn(x, jaw)
