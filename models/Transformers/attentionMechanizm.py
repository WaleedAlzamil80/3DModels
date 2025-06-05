import torch
import torch.nn as nn
from ..GraphCNN.DGCNN import EdgeConv

class LBRD(nn.Module):
    def __init__(self, dim_in=128, dim_out=128, drop_out=0):
        super(LBRD, self).__init__()
        self.L = nn.Conv1d(dim_in, dim_out, 1, bias=False)
        self.B = nn.BatchNorm1d(dim_out)
        self.R = nn.LeakyReLU()
        self.D = nn.Dropout(drop_out)

    def forward(self, x): # (B, N, De) -> (B, N, De)
        return self.D(self.R(self.B(self.L(x.permute(0, 2, 1))))).permute(0, 2, 1)

class LD(nn.Module):
    def __init__(self, dim_in=128, dim_out=128, drop_out=0):
        super(LD, self).__init__()
        self.L = nn.Conv1d(dim_in, dim_out, 1, bias=False)
        self.D = nn.Dropout(drop_out)

    def forward(self, x): # (B, N, De) -> (B, N, De)
        return self.D(self.L(x.permute(0, 2, 1))).permute(0, 2, 1)

class LBD(nn.Module):
    def __init__(self, dim_in=128, dim_out=128, drop_out=0):
        super(LBD, self).__init__()
        self.L = nn.Conv1d(dim_in, dim_out, 1, bias=False)
        self.B = nn.BatchNorm1d(dim_out)
        self.D = nn.Dropout(drop_out)

    def forward(self, x): # (B, N, De) -> (B, N, De)
        return self.D(self.B(self.L(x.permute(0, 2, 1)))).permute(0, 2, 1)

class LRD(nn.Module):
    def __init__(self, dim_in=128, dim_out=128, drop_out=0):
        super(LRD, self).__init__()
        self.L = nn.Conv1d(dim_in, dim_out, 1, bias=False)
        self.R = nn.LeakyReLU()
        self.D = nn.Dropout(drop_out)

    def forward(self, x): # (B, N, De) -> (B, N, De)
        return self.D(self.R(self.L(x.permute(0, 2, 1)))).permute(0, 2, 1)



class PositionEmbedding(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(PositionEmbedding, self).__init__()
        self.LBR1 = LBRD(inchannels, outchannels)
        self.LBR2 = LBRD(outchannels, outchannels)

    def forward(self, x): # B, N, K, C
        return self.LBR2(self.LBR1(x))


class SOA(nn.Module):
    def __init__(self, inchannels, dk, dv, offset = False):
        super(SOA, self).__init__()
        self.offset = offset
        self.k_conv = nn.Conv1d(inchannels, dk, 1, bias = False)
        self.q_conv = nn.Conv1d(inchannels, dk, 1, bias = False)
        self.v_conv = nn.Conv1d(inchannels, dv, 1, bias = False)
        self.dk = torch.tensor(dk, dtype=torch.float32)

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
            return x_r

        else:
            attention = self.softmax(energy / torch.sqrt(self.dk))
            x_w = torch.bmm(attention, v)                             # (B, N, N) @ (B, N, C) -> (B, N, C)
            return x_w


class MultiHeadAttention(nn.Module):
    def __init__(self, inchannels, dk, dv, num_heads, offset = False):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([
            SOA(inchannels, dk, dv, offset) for _ in range(num_heads)
        ])
        self.projection = LBRD(num_heads*dv, inchannels)

    def forward(self, x):
        return self.projection(torch.cat([head(x) for head in self.heads], dim=2))

class GeometryAwareEncoderBlock(nn.Module):
    def __init__(self, inchannels, dk, dv, factor, num_heads, offset):
        super(GeometryAwareEncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(inchannels, dk, dv, num_heads, offset)
        self.edgeconv = EdgeConv(inchannels, [inchannels])

        self.ffn = nn.Sequential(
            LRD(inchannels*2, inchannels*factor),
            LD(factor*inchannels, inchannels)
        )
        self.bn1 = nn.BatchNorm1d(inchannels)
        self.bn2 = nn.BatchNorm1d(inchannels)

    def forward(self, x):
        x1 = self.bn1((x + self.attention(x)).permute(0, 2, 1)).permute(0, 2, 1)
        x2 = self.edgeconv(x)
        x3 = torch.cat([x1, x2], dim=2)
        return self.bn2((x + self.ffn(x3)).permute(0, 2, 1)).permute(0, 2, 1)

class Encoder(nn.Module):
    def __init__(self, inchannels, dk, dv, factor, num_heads, num_layers, offset=False):
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList([
            GeometryAwareEncoderBlock(inchannels, dk, dv, factor, num_heads, offset) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        return x

class QueryGenerator(nn.Module):
    def __init__(self, in_channels, num_points, out_channels):
        super(QueryGenerator, self).__init__()
        self.embedding = nn.Embedding(32, in_channels)
        self.conv_emb = nn.Conv1d(in_channels, in_channels, 1)
        self.conv1 = LBRD(2*in_channels, 3*num_points)
        self.conv2 = LBRD(2*in_channels + 3, out_channels)

    def forward(self, F, teeth): # B, N, D
        try:
            emb = self.embedding(teeth).unsqueeze(2)
        except Exception as e:
            print("teeth shape:", teeth.shape, " value :", teeth)
            print("teeth dtype:", teeth.dtype)
            print("teeth device:", teeth.device)
            raise e
        emb = self.conv_emb(emb).permute(0, 2, 1)

        F = torch.max(F, dim=1, keepdim=True)[0]
        F = torch.cat([F, emb], dim=2)
        points = self.conv1(F).reshape(F.shape[0], -1, 3)
        Q = torch.cat([points, F.expand(points.size(0), points.size(1), -1)], dim = 2)
        return self.conv2(Q), points

class CA(nn.Module):
    def __init__(self, inchannels, encoder_in, dk, dv):
        super(CA, self).__init__()
        self.q_conv = nn.Conv1d(in_channels=inchannels, out_channels=dk, kernel_size=1, bias=False)
        self.k_conv = nn.Conv1d(in_channels=encoder_in, out_channels=dk, kernel_size=1, bias=False)
        self.v_conv = nn.Conv1d(in_channels=encoder_in, out_channels=dv, kernel_size=1, bias=False)
        self.dk = torch.tensor(dk, dtype=torch.float32)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, x, q): # B, N, D     B, M, 3
        k = self.k_conv(x.permute(0, 2, 1))                             # B, Dk, N
        v = self.v_conv(x.permute(0, 2, 1)).permute(0, 2, 1)            # B, N, Dv
        q = self.q_conv(q.permute(0, 2, 1)).permute(0, 2, 1)            # B, M, Dk

        attnetion = self.softmax(torch.bmm(q, k)/torch.sqrt(self.dk))   # B, M, N
        output = torch.bmm(attnetion, v)                                # B, M, Do

        return output

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, inchannels, encoder_in, dk, dv, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.heads = nn.ModuleList([
            CA(inchannels, encoder_in, dk, dv) for _ in range(num_heads)
        ])
        self.projection = LBRD(num_heads*dv, inchannels)

    def forward(self, x, q):
        return self.projection(torch.cat([head(x, q) for head in self.heads], dim = 2))


class DecoderBlock(nn.Module):
    def __init__(self, inchannels, encoder_in, dk, dv, factor, num_heads):
        super(DecoderBlock, self).__init__()
        self.attention = MultiHeadAttention(inchannels, dk, dv, num_heads)
        self.crossattention = MultiHeadCrossAttention(inchannels, encoder_in, dk, dv, num_heads)
        self.edgeconv = EdgeConv(inchannels, [inchannels])

        self.ffn = nn.Sequential(
            LRD(inchannels*2, inchannels*factor),
            LD(factor*inchannels, inchannels)
        )

        self.bn1 = nn.BatchNorm1d(inchannels)
        self.bn2 = nn.BatchNorm1d(inchannels)
        self.bn3 = nn.BatchNorm1d(inchannels)

    def forward(self, x, encoder_input):
        x = self.bn1((x + self.attention(x)).permute(0, 2, 1)).permute(0, 2, 1)
        x1 = self.bn2((x + self.crossattention(encoder_input, x)).permute(0, 2, 1)).permute(0, 2, 1)
        x2 = self.edgeconv(x)
        x3 = torch.cat([x1, x2], dim=2)
        return self.bn3((x + self.ffn(x3)).permute(0, 2, 1)).permute(0, 2, 1)


class Decoder(nn.Module):
    def __init__(self, inchannels, encoder_in, dk, dv, factor, num_heads, num_layers):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(inchannels, encoder_in, dk, dv, factor, num_heads) for _ in range(num_layers)
        ])

    def forward(self, x, encoder_input):
        for layer in self.layers:
            x = layer(x, encoder_input)
        return x
