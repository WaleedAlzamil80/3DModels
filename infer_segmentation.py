import os

import torch
import numpy as np
import fastmesh as fm
import trimesh
import json

from factories.model_factory import get_model
from vis.visulizeGrouped import visualize_with_trimesh
from config.args_config import parse_args
from sklearn.metrics import accuracy_score
from metrics.meanAccClass import compute_mean_per_class_accuracy
from metrics.mIOU import compute_mIoU
from segment import segment
from prepare_vertices import preprocess

args = parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

vertices = fm.load(args.path)[0]
if args.test:
    with open(args.test_ids, 'r') as f: 
            file = json.load(f)
    labels = np.maximum(0, np.array(file['labels']) - 10 - 2 * ((np.array(file['labels']) // 10) - 1))
else:
     labels=None
vertices, labels = preprocess(vertices_np=vertices, cetroids=args.n_centroids, knn=args.nsamples,clean=args.clean, sample=args.sample, labels=labels)

model = get_model(args.model, mode="segmentation", k=33).to(device)
# Load pretrained weights if provided
if args.pretrained and os.path.exists(args.pretrained):
    try:
        print(f"Loading pretrained model from {args.pretrained}")
        state_dict = torch.load(args.pretrained, map_location=device)

        # Strip "module." prefix if the model was saved with DataParallel
        state_dict = {key[7:] if key.startswith("module.") else key: value for key, value in state_dict.items()}
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Failed to load pretrained model from {args.pretrained}. Error: {e}")
        print(f"Ensure the checkpoint matches the architecture '{args.model_name}'.")
else:
    print(f"No pretrained weights found. Initializing a new '{args.model_name}' model.")

model.eval()
output = segment(vertices.to(device), torch.tensor(args.p % 2, dtype=torch.long, device=device).reshape(-1).to(device), model, args.model)

if args.test:
    accPC = compute_mean_per_class_accuracy(output.cuda(), labels.cuda(), 33)
    mIOU = compute_mIoU(output.cuda(), labels.cuda(), 33)
    acc = accuracy_score(labels.view(-1).cpu().numpy(), output.view(-1).cpu().numpy())

    print(f"Accuracy : {acc:.4f}")
    print(f"AccPerCl : {accPC:.4f}")
    print(f"mIOU : {mIOU:.4f}")

# Visulize the performance
if args.visualize:
    visualize_with_trimesh(vertices[0].cpu().detach(), output[0])
