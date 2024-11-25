import torch
from torch import nn

# the input shape should be (Batch_Size, In_channels, Centroids, Samples)
 
class TNetkd(nn.Module):
    def __init__(self, input = 3, mlp = [64, 128, 1024, 512, 256], mode = None, k = None):
        super(TNetkd, self).__init__()
        self.input=input
        self.conv1 = nn.Conv2d(in_channels=self.input, out_channels=mlp[0], kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=mlp[0], out_channels=mlp[1], kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=mlp[1], out_channels=mlp[2], kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=mlp[2], out_channels=mlp[3], kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(in_channels=mlp[3], out_channels=mlp[4], kernel_size=1, bias=False)
        self.conv6 = nn.Conv2d(in_channels=mlp[4], out_channels=self.input*self.input, kernel_size=1)

        self.bn1 = nn.BatchNorm2d(num_features=mlp[0])
        self.bn2 = nn.BatchNorm2d(num_features=mlp[1])
        self.bn3 = nn.BatchNorm2d(num_features=mlp[2])
        self.bn4 = nn.BatchNorm2d(num_features=mlp[3])
        self.bn5 = nn.BatchNorm2d(num_features=mlp[4])

        self.relu = nn.ReLU()

    def forward(self, x):
        bs, _, c, n = x.shape
        x = self.relu(self.bn1(self.conv1(x)))                                          
        x = self.relu(self.bn2(self.conv2(x)))                                          
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

class PointNetGfeat(nn.Module):
    def __init__(self, input = 3, k=33, mlp = [64, 64, 64, 128, 1024]):
        super(PointNetGfeat, self).__init__()
        self.input = input
        self.Tnet3d = TNetkd(self.input)
        self.conv1 = nn.Conv2d(in_channels=self.input, out_channels=mlp[0], kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=mlp[0], out_channels=mlp[1], kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mlp[0])
        self.bn2 = nn.BatchNorm2d(mlp[1])

        self.Tnet64d = TNetkd(mlp[1])
        self.conv3 = nn.Conv2d(mlp[1], mlp[2], 1, bias=False)
        self.conv4 = nn.Conv2d(mlp[2], mlp[3], 1, bias=False)
        self.conv5 = nn.Conv2d(mlp[3], mlp[4], 1, bias=False)
        self.bn3 = nn.BatchNorm2d(mlp[2])
        self.bn4 = nn.BatchNorm2d(mlp[3])
        self.bn5 = nn.BatchNorm2d(mlp[4])

        self.relu = nn.ReLU()

    def forward(self, x):                                                   # x.shape = # (B, k, C, #pointClouds)
        bs, _, c, n = x.shape
        inT = self.Tnet3d(x)
        x = torch.bmm(x.reshape(bs, -1, self.input), inT).view(bs, self.input, c, n)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        feT = self.Tnet64d(x)
        local_features = torch.bmm(x.view(bs, -1, 64), feT).view(bs, 64, c, n)
        x = self.relu(self.bn3(self.conv3(local_features)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))

        global_features = torch.max(x, dim = 3, keepdim=True)[0]
        global_features = torch.max(global_features, dim = 2, keepdim=True)[0]

        x = torch.cat([local_features, global_features.expand(-1, -1, c, n)], dim = 1)
        return x, global_features, inT, feT

class PointNetCls(nn.Module):
    def __init__(self, k = 28, input=3, mlp=[1024,512,256]):
        super(PointNetCls, self).__init__()
        self.k = k
        self.input = input
        self.feNet = PointNetGfeat(k = self.input)
        self.fc1 = nn.Conv2d(mlp[0], mlp[1], 1, bias=False)
        self.fc2 = nn.Conv2d(mlp[1], mlp[2], 1, bias=False)
        self.fc3 = nn.Conv2d(mlp[2], self.k, 1)

        self.bn1 = nn.BatchNorm2d(mlp[1])
        self.bn2 = nn.BatchNorm2d(mlp[2])
        self.drop25 = nn.Dropout(0.25)
        self.drop70 = nn.Dropout(0.70)

        self.relu = nn.ReLU()

    def forward(self, x):
        _, x, inTra, feTra = self.feNet(x)         # (b, 1024, 1, 1)

        x = self.relu(self.drop25(self.bn1(self.fc1(x))))
        x = self.relu(self.drop70(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = x.squeeze(3).squeeze(2)

        return x, inTra, feTra

class PointNetSeg(nn.Module):
    def __init__(self, k = 28, input = 3, mlp=[1088, 512, 256, 128]):
        super(PointNetSeg, self).__init__()
        self.input = input
        self.k = k
        self.feNet = PointNetGfeat(k = self.input)
        self.conv1 = nn.Conv2d(mlp[0], mlp[1], 1, bias=False)
        self.conv2 = nn.Conv2d(mlp[1], mlp[2], 1, bias=False)
        self.conv3 = nn.Conv2d(mlp[2], mlp[3], 1, bias=False)
        self.conv4 = nn.Conv2d(mlp[3], self.k, 1)
        self.bn1 = nn.BatchNorm2d(mlp[1])
        self.bn2 = nn.BatchNorm2d(mlp[2])
        self.bn3 = nn.BatchNorm2d(mlp[3])

        self.relu = nn.ReLU()

    def forward(self, x):
        x, _, inTra, feTra = self.feNet(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)

        return x, inTra, feTra


# Mode Factory that maps modes to classes
MODE_FACTORY = {
    "classification": PointNetCls,
    "segmentation": PointNetSeg,
    "features": PointNetGfeat,
}

def get_pointnet_mode(mode, *args, **kwargs):
    """Fetch the appropriate PointNet model based on the mode."""
    if mode not in MODE_FACTORY:
        raise ValueError(f"Mode {mode} is not available.")
    return MODE_FACTORY[mode](*args, **kwargs)


class PointNet(nn.Module):
    def __init__(self, mode = "classification", k=28, input=3):
        super(PointNet, self).__init__()
        self.PointNet = get_pointnet_mode(mode, k = k, input = input)

    def forward(self, x):
        # If input is (Batch_size, #channels, #points), we need to reshape it to (Batch_size, #channels, 1, #points)
        if len(x.shape) == 3:
            bs, channels, points = x.shape
            x = x.transpose(1, 2).unsqueeze(2)

        return self.PointNet(x)

class PointNetPartSeg(nn.Module):
    def __init__(self, k=33, input = 3):
        super(PointNetPartSeg, self).__init__()

    def forward(self, x):
        return x
