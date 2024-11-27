import torch
import torch.nn as nn
from models.FoldingNet.Mining import GaussianKernelConv
from models.GraphCNN.DGCNN import EdgeConv
from sampling.PointsCloud.knn import kdneighGPU

import argparse

class TNetkd(nn.Module):
    def __init__(self, input = 3, mlp = [64, 128, 1024, 512, 256], mode = None, k = 32):
        super(TNetkd, self).__init__()
        self.kc = GaussianKernelConv(input, mlp[1], sigma=1.0)
        self.args = argparse.Namespace(**{'knn': k})

        self.input=input
        self.edgeconv1 = EdgeConv(self.input, [mlp[0], mlp[0]])
        self.edgeconv2 = EdgeConv(mlp[0], [mlp[1], mlp[1]])
        self.edgeconv3 = EdgeConv(2*mlp[1], [mlp[2], mlp[2]])

        self.conv1 = nn.Conv2d(in_channels=mlp[2], out_channels=mlp[3], kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=mlp[3], out_channels=mlp[4], kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=mlp[4], out_channels=self.input*self.input, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(num_features=mlp[3])
        self.bn2 = nn.BatchNorm2d(num_features=mlp[4])

        self.relu = nn.LeakyReLU()

    def forward(self, x): # (Batch_Size, In_channels, Centroids, Samples)
        bs, _, c, n = x.shape
        xnei = kdneighGPU(x.reshape(bs, -1, c*n).permute(0, 2, 1), self.args)[1]
        kernels = self.kc(xnei)
        x = self.edgeconv1(x.reshape(bs, c*n, -1))
        x = self.edgeconv2(x)
        x = torch.cat([x.reshape(bs, -1, c, n), kernels.permute(0, 2, 1).reshape(bs, -1, c, n)], dim = 1)

        x = self.edgeconv3(x.reshape(bs, c*n, -1)).reshape(bs, -1, c, n)
        x = self.relu(self.bn1(self.conv1(x)))                                       
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)                                                

        x = torch.max(x, dim=3, keepdim=False)[0]                                            # (B, k * k, C)
        x = torch.max(x, dim=2, keepdim=False)[0]                                            # (B, k * k)

        iden = torch.eye(self.input, requires_grad=True).view(1, self.input * self.input).expand(bs, -1).to(x.device)
        x = x + iden
        x = x.view(bs, self.input, self.input)
        return x
