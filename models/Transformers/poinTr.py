import torch
import torch.nn as nn
from models.GraphCNN.DGCNN import EdgeConv
from factories.sampling_factory import get_sampling_technique

import argparse

sampling = get_sampling_technique('fps')

'''
We employ a lightweight DGCNN [44] model to 
extract the point proxy features. To reduce 
the computational cost, we hierarchically downsample 
the original input point cloud to N = 128 center 
points and use several DGCNN layers to capture 
local geometric relationships. The detailed network 
architecture is:
'''


class DGCNN(nn.Module):
    def __init__(self, Cin, Cout, k, Nout, f_embeding = True, *args, **kwargs):
        super(DGCNN, self).__init__(*args, **kwargs)
        self.k, self.Nout = k, Nout

        self.edgeconv1 = EdgeConv(Cin [Cout], k)
        self.edgeconv2 = EdgeConv(Cout [Cout], k)
        self.edgeconv3 = EdgeConv(Cout [Cout], k)
        self.convProj = nn.Conv1d(3*Cout + Cin, Cout, 1)

        self.relu = nn.LeakyReLU()

    def forward(self, x): # B, N, Cin -> # B, Nout, Cout
        ############# Sampling Nout as farthers point Sampling ########################
        x1 = self.edgeconv1(x)
        x2 = self.edgeconv2(x1)
        x3 = self.edgeconv3(x2)
        x = torch.cat([x, x1, x2, x3], dim = -1)
        return self.convProj(x)


class FeatureExtractor(nn.Module):
    def __init__(self, *args, **kwargs):
        super(FeatureExtractor, self).__init__(*args, **kwargs)
        self.embedding = nn.Embedding(2, 64)
        self.conv_emb = nn.Conv1d(64, 256, 1)
        self.Linear = nn.Conv1d(3, 8, 1)

        self.dgcnn1 = DGCNN(Cin = 8, Cout = 32, K = 8, Nout = 2048)
        self.dgcnn2 = DGCNN(Cin = 32, Cout = 64, K = 8, Nout = 512)
        self.dgcnn3 = DGCNN(Cin = 64, Cout = 64, K = 8, Nout = 512)
        self.dgcnn4 = DGCNN(Cin = 64, Cout = 128, K = 8, Nout = 128)

    def forward(self, x):
        return x
