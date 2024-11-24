import os

import torch
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

from SegmentGenerate import SegmentGenerate

vertices = fm.load(args.path)[0]

output = torch.Tensor(SegmentGenerate(vertices, args.p, args.model, args.mode, args.pretrained, True, args.clean))
print(output)
print(output.shape)

# if args.test:
#     with open(args.test_ids, 'r') as f:
#             file = json.load(f)
#     labels = np.maximum(0, np.array(file['labels']) - 10 - 2 * ((np.array(file['labels']) // 10) - 1))
#     print(labels.shape, sum(labels==0))
#     labels = torch.tensor(np.array(labels, dtype=np.int64)[valid_mask][idx], dtype=torch.long).view(1, -1)
# 
#     print(output.device, labels.device)
#     accPC = compute_mean_per_class_accuracy(output.cuda(), labels.cuda(), args.k)
#     mIOU = compute_mIoU(output.cuda(), labels.cuda(), args.k)
#     print(output.view(-1).cpu().numpy())
#     acc = accuracy_score(labels.view(-1).cpu().numpy(), output.view(-1).cpu().numpy())
# 
#     print(f"Accuracy : {acc:.4f}")
#     print(f"AccPerCl : {accPC:.4f}")
#     print(f"mIOU : {mIOU:.4f}")

# visualize_with_trimesh(vertices[0].cpu().detach(), output[0])
