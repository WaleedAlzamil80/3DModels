import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=20):
        super(EdgeConv, self).__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        )