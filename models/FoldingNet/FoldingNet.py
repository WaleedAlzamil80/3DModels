import argparse
import numpy as np
import torch
import torch.nn as nn
from sampling.PointsCloud.knn import knn_neighbors as kdneighGPU, compute_local_covariance

class GraphLayer(nn.Module):
    def __init__(self, in_channels, out_channels, k=32):
        super(GraphLayer, self).__init__()
        self.graphL = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(),
        )
        self.args = argparse.Namespace(**{'knn': k})

    def forward(self, x):  # B, N, C
        x = kdneighGPU(x, self.args)[1]  # B, N, K, C
        x = torch.max(x, dim=2, keepdim=False)[0] # B, N, C
        x = self.graphL(x.permute(0, 2, 1)) # B, C, N

        return x.permute(0, 2, 1) # B, N, C


class GBEncoder(nn.Module):
    def __init__(self, infeatures, outfeatures, k = 32):
        super(GBEncoder, self).__init__() 
        self.conv1 = nn.Conv1d(infeatures, outfeatures, 1, bias=False)
        self.conv2 = nn.Conv1d(outfeatures, outfeatures, 1, bias=False)
        self.conv3 = nn.Conv1d(outfeatures, outfeatures, 1, bias=False)
        self.conv4 = nn.Conv1d(1024, 512, 1, bias=False) # for the codeword
        self.bn1 = nn.BatchNorm1d(outfeatures)
        self.bn2 = nn.BatchNorm1d(outfeatures)
        self.bn3 = nn.BatchNorm1d(outfeatures)
        self.bn4 = nn.BatchNorm1d(512) # for the codeword
        self.gl1 = GraphLayer(64, 128)
        self.gl2 = GraphLayer(128, 1024)
        self.relu = nn.ReLU()
        self.args = argparse.Namespace(**{'knn': k})

    def forward(self, x):

        x = torch.cat([x, compute_local_covariance(kdneighGPU(x, self.args)[1])], dim = 2).permute(0, 2, 1) # B, C, N
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.gl1(x.permute(0, 2, 1))
        x = self.gl2(x)                # B, N, C
        x = torch.max(x, dim=1,keepdim=True)[0] # B, 1, C

        return self.relu(self.bn4(self.conv4(x.permute(0, 2, 1)))).permute(0, 2, 1)
 

class Folding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Folding, self).__init__()

        self.fold = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels[0], kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels[0]),
            nn.LeakyReLU(),
        )
        for i in range(1, len(out_channels)):
            self.fold.append(nn.Sequential(
            nn.Conv1d(in_channels=out_channels[i-1], out_channels=out_channels[i], kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels[i]),
            nn.LeakyReLU(),
        ))

    def forward(self, x):
        return self.fold(x)


class FBDecoder(nn.Module):
    def __init__(self, num_points=32):
        super(FBDecoder, self).__init__() 
        # Sample the grids in 2D space
        self.grid = torch.tensor(np.array(np.meshgrid(np.linspace(-0.5, 0.5, num_points, dtype=np.float32), np.linspace(-0.5, 0.5, num_points, dtype=np.float32))), dtype=torch.float32).reshape(-1, 2)
        self.fold1 = Folding(514, (512, 512, 3))
        self.fold2 = Folding(515, (512, 512, 3))

    def forward(self, codeword):  # B, 1, C
        codeword = codeword.expand(-1, self.grid.shape[0], -1) # B, M, C
        x = torch.cat([codeword, self.grid.to(codeword.device).unsqueeze(0).expand(codeword.shape[0], -1, -1)], dim = 2) # B, M, Cnew
        x = self.fold1(x.permute(0, 2, 1)).permute(0, 2, 1) # B, M, 3
        x = torch.cat([x, codeword], dim = 2) # B, M, Cnew2
        x = self.fold2(x.permute(0, 2, 1)).permute(0, 2, 1) # B, M, 3

        return x


class FoldingNet(nn.Module):
    def __init__(self, encoder_in=12, encoder_out=64, num_points=32):
        super(FoldingNet, self).__init__()
        self.gbencoder = GBEncoder(encoder_in, encoder_out)
        self.fbdecoder = FBDecoder(num_points)

    def forward(self, x):
        codeword = self.gbencoder(x)
        repoints = self.fbdecoder(codeword)

        return repoints
