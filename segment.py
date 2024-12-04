import torch

def segment(vertices, jaw, model, model_name):

    if model_name != "DynamicGraphCNN":
        output = model(vertices, jaw)
    else:
        output = model(vertices, jaw)[0]
    output = torch.max(output, dim=2)[1].cpu().detach()

    return output
