import torch
import torch.nn as nn
from GraphCNN.DGCNN import DGCNN
from Transformers.attentionMechanizm import PositionEmbedding, Encoder, Decoder, QueryGenerator

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.posembedding = PositionEmbedding(3, 128)
        self.dgcnn = DGCNN()

    def forward(self, x, jaw):
        le = self.dgcnn(x, jaw)
        pe = self.posembedding(x)
        return torch.cat([pe, le], dim = 2)

class TransformerArchitecture(nn.Module):
    def __init__(self, in_channels=256, num_points=512, out_channels=256):
        super(TransformerArchitecture, self).__init__()
        self.encoder = Encoder(in_channels, dk=512, dv=512, factor=4, num_heads=4, num_layers=2)
        self.decoder = Decoder(in_channels, encoder_in=256, dk=512, dv=512, factor=4, num_heads=4, num_layers=3)
        self.QG = QueryGenerator(in_channels, num_points, out_channels)

    def forward(self, x, teeth):
        encoder_in = self.encoder(x)
        # print("Encoder output shape:", encoder_in.shape)
        query, points = self.QG(x, teeth)
        # print("Query shape:", query.shape)
        # print("Points shape:", points.shape)
        decoder_out = self.decoder(query, encoder_in)
        # print("Decoder output shape:", decoder_out.shape)
        return decoder_out, points

class Model(nn.Module):
    def __init__(self, inchannels=256, dk=256, dv=256, factor=4, num_heads=4, num_layers=2, num_points = 512, offset=False):
        super(Model, self).__init__()
        self.features = FeatureExtractor()
        self.transformer = TransformerArchitecture() # (inchannels=inchannels, dk=dk, dv=dv, factor=factor, num_heads=num_heads, num_layers=num_layers, offset=offset)
        self.projection = nn.Conv1d(in_channels=256+3, out_channels=3, kernel_size=1)
        # self.foldingnet = FoldingNet(num_points=num_points)

    def forward(self, x, teeth, jaw):
        x = self.features(x, jaw)
        x, points = self.transformer(x, teeth)
        x = torch.cat([x, points], dim = 2)
        x = self.projection(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x
        # fol = self.foldingnet(x)
        # return fol +  points
