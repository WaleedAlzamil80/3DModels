import torch
import torch.nn as nn
import argparse
from sampling.PointsCloud.knn import kdneighGPU


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

class PCTEncoder(nn.Module):
    def __init__(self, dim_in=3, dim_embed=128, globalFeatures=1024):
        super(PCTEncoder, self).__init__()
        self.embedding = NeighborEmbedding(dim_in, dim_embed)
        self.oa1 = OA(2*dim_embed, dim_embed//4)
        self.oa2 = OA(2*dim_embed, dim_embed//4)
        self.oa3 = OA(2*dim_embed, dim_embed//4)
        self.oa4 = OA(2*dim_embed, dim_embed//4)
        self.lbr1 = LBRD(dim_embed*8, globalFeatures)

    def forward(self, x):
        x = self.embedding(x)
        oa1 = self.oa1(x)
        oa2 = self.oa2(oa1)
        oa3 = self.oa3(oa2)
        oa4 = self.oa4(oa3)
        x = torch.concat([oa1, oa2, oa3, oa4], dim = 2)
        x = self.lbr1(x.unsqueeze(2)).squeeze(2)                                   # Point Features

        return x


class PCTclassification(nn.Module):
    def __init__(self, dim_in=3, dim_embed=128, globalFeatures=1024, k=33):
        super(PCTclassification, self).__init__()
        self.encoder = PCTEncoder(dim_in, dim_embed, globalFeatures)
        self.lbrd1 = LBRD(globalFeatures*2, dim_embed*2, 0.5)
        self.lbrd2 = LBRD(dim_embed*2, dim_embed*2, 0.5)
        self.linear = nn.Conv2d(dim_embed*2, k, 1)

    def forward(self, x):
        x = self.encoder(x)
        gfmax = torch.max(x, dim=1, keepdim=False) # B, D
        gfmean = torch.mean(x, dim=1, keepdim=False)
        gf = torch.cat([gfmax, gfmean], dim=1)         # B, GF
        x = self.lbrd1(gf.unsqueeze(1).unsqueeze(2))   # B, 1, 1, GF
        x = self.lbrd2(x)
        x = self.linear(x.permute(0, 3, 2, 1)).squeeze(3).squeeze(2) # B, Nc

        return x


class PCTsegmentation(nn.Module):
    def __init__(self, dim_in=3, dim_embed=128, k=33, globalFeatures=1024, dim_cat = 64):
        super(PCTsegmentation, self).__init__()
        self.encoder = PCTEncoder(dim_in, dim_embed, globalFeatures)
        self.embedding = nn.Embedding(2, dim_cat)
        self.embed_conv = nn.Conv1d(dim_cat, dim_cat, 1)

        self.lbrd1 = LBRD(globalFeatures*3 + dim_cat, dim_embed*2, 0.5)
        self.lbrd2 = LBRD(dim_embed*2, dim_embed*2)
        self.linear = nn.Conv2d(dim_embed*2, k, 1)

    def forward(self, x, category):
        x = self.encoder(x)                            # B, N, D
        cat = self.embedding(category).unsqueeze(2)    # B, 64, 1
        cat = self.embed_conv(cat)

        gfmax = torch.max(x, dim=1, keepdim=True)[0]      # B, 1, D
        gfmean = torch.mean(x, dim=1, keepdim=True)
        gf = torch.cat([gfmax, gfmean, cat.permute(0, 2, 1)], dim=2)         # B, 1, 2D + 64
        x = torch.concat([x, gf.expand(-1, x.shape[1], -1)], dim=2)    # B, N, 3D + 64

        x = self.lbrd1(x.unsqueeze(2))
        x = self.lbrd2(x)
        x = self.linear(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1).squeeze(2) # B, N, k

        return x


# Mode Factory that maps modes to classes
MODE_FACTORY = {
    "classification": PCTclassification,
    "segmentation": PCTsegmentation,
}

def get_pct_mode(mode, *args, **kwargs):
    """Fetch the appropriate PCT model based on the mode."""
    if mode not in MODE_FACTORY:
        raise ValueError(f"Mode {mode} is not available.")
    return MODE_FACTORY[mode](*args, **kwargs)

class PCTransformer(nn.Module):
    def __init__(self, *args, **kwargs):
        super(PCTransformer, self).__init__()
        self.pct = get_pct_mode(*args, **kwargs)

    def forward(self, x, jaw):
        return self.pct(x, jaw)
