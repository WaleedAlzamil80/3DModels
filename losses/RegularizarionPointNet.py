import torch

def tnet_regularization(transforms):
    bs = transforms.shape[0]
    d = transforms.shape[1]
    I = torch.eye(d, requires_grad=True).view(1, d, d).repeat(bs, 1, 1)

    if transforms.is_cuda:
        I = I.cuda()
    diff = I - torch.bmm(transforms, transforms.transpose(1, 2))

    return torch.norm(diff, p = 2, dim=(1, 2)).mean()

def project_to_so3(matrix):
    U, _, Vt = torch.linalg.svd(matrix)
    R = torch.matmul(U, Vt)
    # Fix improper rotation if needed
    if torch.det(R) < 0:
        U[:, -1] *= -1
        R = torch.matmul(U, Vt)
    return R
