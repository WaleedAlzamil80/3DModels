import torch
import torch.nn as nn
import argparse
from sampling.PointsCloud.knn import knn_neighbors as kdneighGPU

class LBRD(nn.Module):
    def __init__(self, dim_in=128, dim_out=128, drop_out=0):
        super(LBRD, self).__init__()
        self.L = nn.Conv2d(dim_in, dim_out, 1, bias=False)
        self.B = nn.BatchNorm2d(dim_out)
        self.R = nn.LeakyReLU()
        self.D = nn.Dropout(drop_out)

    def forward(self, x): # (B, N, k, De) -> (B, N, k, De)
        return self.D(self.R(self.B(self.L(x.permute(0, 3, 2, 1))))).permute(0, 3, 2, 1)


class EmbeddingInput(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(EmbeddingInput, self).__init__()
        self.LBR1 = LBRD(inchannels, outchannels)
        self.LBR2 = LBRD(outchannels, outchannels)

    def forward(self, x): # B, N, K, C
        return self.LBR2(self.LBR1(x))


class NeighborEmbedding(nn.Module):
    def __init__(self, inchannels, outchannels, k=32):
        super(NeighborEmbedding, self).__init__()
        self.embed1 = EmbeddingInput(inchannels, outchannels)
        self.embed2 = EmbeddingInput(2*outchannels, 2*outchannels)

        self.args = argparse.Namespace(**{'knn': k})

    def forward(self, x):                                         # B, N, C
        x = self.embed1(x.unsqueeze(2)).squeeze(2)
        x = kdneighGPU(x, self.args)[0]                           # B, N, k, 2D

        x = self.embed2(x)                                        # B, N, k, 2D
        x = torch.max(x, dim=2, keepdim=False)[0]                 # B, N, Do

        return x


class OA(nn.Module):
    def __init__(self, inchannels, outchannels, offset = True):
        super(OA, self).__init__()
        self.offset = offset
        self.k_conv = nn.Conv1d(inchannels, outchannels, 1, bias = False)
        self.q_conv = nn.Conv1d(inchannels, outchannels, 1, bias = False)
        self.v_conv = nn.Conv1d(inchannels, inchannels, 1, bias = False)
        self.project = LBRD(inchannels, inchannels)
        if self.offset:
            self.softmax = nn.Softmax(dim = 1)
        else:
            self.softmax = nn.Softmax(dim = -1)

    def forward(self, x):                                         # B, N, C
        q = self.q_conv(x.permute(0, 2, 1)).permute(0, 2, 1)      # B, N, D
        k = self.k_conv(x.permute(0, 2, 1))                       # B, D, N
        v = self.v_conv(x.permute(0, 2, 1)).permute(0, 2, 1)      # B, N, C
        energy = torch.bmm(q, k)                                  # B, N, N

        if self.offset:
            attention = self.softmax(energy)
            attention = attention / (1e-9 + attention.sum(dim=-1, keepdims=True))
            x_r = torch.bmm(attention, v)                             # (B, N, N) @ (B, N, C) -> (B, N, C)
            x_r = self.project((x - x_r).unsqueeze(2)).squeeze(2)
        else:
            attention = self.softmax(energy / torch.sqrt(q.shape[2]))
            x_w = torch.bmm(attention, v)                             # (B, N, N) @ (B, N, C) -> (B, N, C)
            x_w = self.project(x_w)

        return x_r + x

#################################################### MHS\OA ####################################################

class attentionTNET(nn.Module):
    def __init__(self, dim_in=3, dim_embed=128, globalFeatures=1024, mode = None, k = 22):
        super(attentionTNET, self).__init__()
        self.input = dim_in
        self.embedding = NeighborEmbedding(dim_in, dim_embed)
        self.oa1 = OA(2*dim_embed, dim_embed//4)
        self.oa2 = OA(2*dim_embed, dim_embed//4)
        self.oa3 = OA(2*dim_embed, dim_embed//4)
        self.oa4 = OA(2*dim_embed, dim_embed//4)
        self.lbr1 = LBRD(dim_embed*8, globalFeatures)
        self.conv = nn.Conv1d(in_channels=globalFeatures, out_channels=self.input*self.input, kernel_size=1, bias=False)

    def forward(self, x):
        bs = x.shape[0]
        x = self.embedding(x)
        oa1 = self.oa1(x)
        oa2 = self.oa2(oa1)
        oa3 = self.oa3(oa2)
        oa4 = self.oa4(oa3)
        x = torch.concat([oa1, oa2, oa3, oa4], dim = 2)
        x = self.lbr1(x.unsqueeze(2)).squeeze(2)                                   # Point Features B N D
        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)          # B N 9                                                      
        x = torch.max(x, dim=1, keepdim=False)[0]                                            # (B, k * k)

        iden = torch.eye(self.input, requires_grad=True).view(1, self.input * self.input).expand(bs, -1).to(x.device)
        x = x + iden
        x = x.view(bs, self.input, self.input)

        return x
