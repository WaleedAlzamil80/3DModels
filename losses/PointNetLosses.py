import torch

def tnet_regularization(transforms):
    bs = transforms.shape[0]
    d = transforms.shape[1]
    I = torch.eye(d, requires_grad=True).view(1, d, d).repeat(bs, 1, 1)

    if transforms.is_cuda:
        I = I.cuda()
    diff = I - torch.bmm(transforms, transforms.transpose(1, 2))

    return torch.norm(diff, p = 2, dim=(1, 2)).mean()