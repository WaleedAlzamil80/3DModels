import torch
import torch.nn as nn
from models.FoldingNet.Mining import GaussianKernelConv
from sampling.PointsCloud.knn import kdneighGPU
import argparse

class LBRD(nn.Module):
    def __init__(self, dim_in=128, dim_out=128, drop_out=0):
        super(LBRD, self).__init__()
        self.L = nn.Conv2d(dim_in, dim_out, 1, bias=False)
        self.B = nn.BatchNorm2d(dim_out)
        self.R = nn.ReLU()
        self.D = nn.Dropout(drop_out)

    def forward(self, x): # (B, De, k, N) -> (B, De, k, N)
        return self.D(self.R(self.B(self.L(x))))

class TNetkd(nn.Module):
    def __init__(self, input = 3, mlp = [64, 128, 1024, 512, 256], mode = None, k = 32):
        super(TNetkd, self).__init__()
        self.kc = GaussianKernelConv(input, mlp[1], sigma=1.0)
        self.args = argparse.Namespace(**{'knn': k})

        self.input=input
        self.conv1 = nn.Conv2d(in_channels=self.input, out_channels=mlp[0], kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=mlp[0], out_channels=mlp[1], kernel_size=1, bias=False)
    
        self.conv3 = nn.Conv2d(in_channels=2*mlp[1], out_channels=mlp[2], kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=mlp[2], out_channels=mlp[3], kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(in_channels=mlp[3], out_channels=mlp[4], kernel_size=1, bias=False)
        self.conv6 = nn.Conv2d(in_channels=mlp[4], out_channels=self.input*self.input, kernel_size=1)

        self.bn1 = nn.BatchNorm2d(num_features=mlp[0])
        self.bn2 = nn.BatchNorm2d(num_features=mlp[1]) 
        self.bn3 = nn.BatchNorm2d(num_features=mlp[2])
        self.bn4 = nn.BatchNorm2d(num_features=mlp[3])
        self.bn5 = nn.BatchNorm2d(num_features=mlp[4])

        self.relu = nn.ReLU()

    def forward(self, x): # (Batch_Size, In_channels, Centroids, Samples)
        bs, _, c, n = x.shape
        xnei = kdneighGPU(x.reshape(bs, -1, c*n).permute(0, 2, 1), self.args)[1]
        kernels = self.kc(xnei)
        x = self.relu(self.bn1(self.conv1(x)))                                          
        x = self.relu(self.bn2(self.conv2(x)))
        x = torch.cat([x, kernels.permute(0, 2, 1).reshape(bs, -1, c, n)], dim = 1)
        x = self.relu(self.bn3(self.conv3(x)))                                                
        x = self.relu(self.bn4(self.conv4(x)))                                                
        x = self.relu(self.bn5(self.conv5(x)))                                                
        x = self.conv6(x)                                                                     

        x = torch.max(x, dim=3, keepdim=False)[0]                                            # (B, k * k, C)
        x = torch.max(x, dim=2, keepdim=False)[0]                                            # (B, k * k)

        iden = torch.eye(self.input, requires_grad=True).view(1, self.input * self.input).expand(bs, -1).to(x.device)
        x = x + iden
        x = x.view(bs, self.input, self.input)
        return x
