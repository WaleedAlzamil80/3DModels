import sys
sys.path.append('/home/waleed/Documents/3DLearning/3DModels')
import os

import torch
from torch import nn
import numpy as np
import fastmesh as fm
import json

from factories.model_factory import get_model
from factories.sampling_factory import get_sampling_technique
from vis.visulizeGrouped import visualize_with_trimesh
from config.args_config import parse_args
from sklearn.metrics import accuracy_score
from metrics.meanAccClass import compute_mean_per_class_accuracy
from metrics.mIOU import compute_mIoU

args = parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Use the factory to dynamically get the model
model = get_model(args.model, mode=args.mode, k=args.k).to(device)
model = nn.DataParallel(model).to(device)

if args.pretrained is not None and os.path.exists(args.pretrained):
    try:
        print(f"Loading pretrained model from {args.pretrained}")
        state_dict = torch.load(args.pretrained, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    except:
        print(f"The pretrained model {args.pretrained} is not for {args.model} architecture")
else:
    print(f"Instantiating new model from {args.model}")

model.eval()

vertices_np = fm.load(args.path)[0] # np.array(mesh.vertices)
if args.clean:
    origin = np.mean(vertices_np, axis=0)

    z_values = vertices_np[:, 2]
    y_values = vertices_np[:, 1]
    x_values = vertices_np[:, 0]

    y_mean = np.mean(y_values)
    y_std = np.std(y_values)
    x_mean = np.mean(x_values)
    x_std = np.std(x_values)
    alpha = 2.0

    valid_mask = (z_values > (origin[2] - 5)) & \
                    (y_values < (y_mean + alpha * y_std)) & (y_values > (y_mean - alpha * y_std)) & \
                    (x_values < (x_mean + alpha * x_std)) & (x_values > (x_mean - alpha * x_std))

    vertices_np = vertices_np[valid_mask]

sampling = get_sampling_technique(args.sampling)
vertices_np, idx = sampling(vertices_np, args.n_centroids, args.nsamples)
vertices = torch.tensor(vertices_np, dtype=torch.float32, device=device).view(-1, 3).unsqueeze(0)
jaw = torch.tensor(args.p % 2, dtype=torch.long, device=device).reshape(-1)

output = model(vertices, jaw)[0]
output = torch.max(output, dim=2, keepdim=False)[1].cpu().detach()

if args.test:
    with open(args.test_ids, 'r') as f:
            file = json.load(f)
    labels = np.maximum(0, np.array(file['labels']) - 10 - 2 * ((np.array(file['labels']) // 10) - 1))
    labels = torch.tensor(np.array(labels, dtype=np.int64)[valid_mask][idx], dtype=torch.long).view(1, -1)

    print(output.device, labels.device)
    accPC = compute_mean_per_class_accuracy(output.cuda(), labels.cuda(), args.k)
    mIOU = compute_mIoU(output.cuda(), labels.cuda(), args.k)
    print(output.view(-1).cpu().numpy())
    acc = accuracy_score(labels.view(-1).cpu().numpy(), output.view(-1).cpu().numpy())

    print(f"Accuracy : {acc:.4f}")
    print(f"AccPerCl : {accPC:.4f}")
    print(f"mIOU : {mIOU:.4f}")

# visualize_with_trimesh(vertices[0].cpu().detach(), output[0])
