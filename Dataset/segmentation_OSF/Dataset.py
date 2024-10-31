import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
import numpy as np
import fastmesh as fm
from factories.sampling_factory import get_sampling_technique
from ..transformations.rigid import random_rigid_transform

class TeethSegmentationDataset(Dataset):
    def __init__(self, split='train',  transform=None, p=7, args = None):
        """
        Args:
            root_dir (string): Directory with all the parts (data_part_{1-6}).
            split (string): 'train' or 'test' to select the appropriate dataset.
            test_ids_file (string): Path to the txt file containing IDs for testing.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.jaw_to_idx = {"lower": 0, "upper": 1}
        self.args = args
        self.split = split
        self.transform = transform
        self.p = p

        self.sampling_fn = get_sampling_technique(args.sampling)
        self.test_ids = self._load_test_ids(args.test_ids)
        self.data_list = self._prepare_data_list()

    def _load_test_ids(self, test_ids_file):
        """Load IDs from private-testing-set.txt."""
        with open(test_ids_file, 'r') as f:
            ids = [line.strip() for line in f.readlines()]
        return ids

    def _prepare_data_list(self):
        """Prepare the list of data paths for training or testing."""
        data_list = []
        for part in range(1, self.p + 1):
            part_dir = os.path.join(self.args.path, f'data_part_{part}')
            for region in ['lower', 'upper']:
                region_dir = os.path.join(part_dir, region)
                for sample_id in os.listdir(region_dir):
                    sr = sample_id + "_" + region
                    if (self.split == 'test' and sr in self.test_ids) or \
                       (self.split == 'train' and sr not in self.test_ids):
                        bmesh_path = os.path.join(region_dir, sample_id, f'{sample_id}_{region}.bmesh')
                        label_path = os.path.join(region_dir, sample_id, f'{sample_id}_{region}.json')
                        data_list.append((bmesh_path, label_path))
        return data_list

    def __len__(self):
        return len(self.data_list)

    def _load_bmesh_file(self, bmesh_path):
        """Load .bemsh file, clean vertices using NumPy, and return processed vertices."""
        # Load the .bemsh file using trimesh
        vertices_np = fm.load(bmesh_path)[0]

        # Step 1: Use NumPy for initial cleaning
        origin = np.mean(vertices_np, axis=0)

        # Apply NumPy filtering on z, y, and x values based on given conditions
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

        # Apply the mask to filter points
        vertices_np_cleaned = vertices_np[valid_mask]

        points, idx = self.sampling_fn(vertices_np_cleaned, self.args.n_centroids, self.args.nsamples)

        return points, idx, valid_mask

    def _load_labels(self, label_path):
        """Load labels from the JSON file."""
        with open(label_path, 'r') as f:
            file = json.load(f)
        labels = np.maximum(0, np.array(file['labels']) - 10 - 2 * ((np.array(file['labels']) // 10) - 1))
        return np.array(labels, dtype=np.int64), torch.tensor(self.jaw_to_idx[file['jaw']], dtype=torch.long)

    def __getitem__(self, idx):
        bmesh_path, label_path = self.data_list[idx]

        vertices, idx, valid_mask = self._load_bmesh_file(bmesh_path)

        if self.transform:
            vertices = random_rigid_transform(vertices)

        # Convert vertices to a PyTorch tensor and apply the view transformation
        vertices = torch.tensor(vertices, dtype=torch.float32).view(-1, 3)

        labels, jaw = self._load_labels(label_path)
        labels = torch.tensor(labels[valid_mask][idx], dtype=torch.long)
        return vertices.view(-1, 3), labels.view(-1), jaw

# Usage of the dataset
def OSF_data_loaders(args):
    # Create training and testing datasets
    train_dataset = TeethSegmentationDataset(split='train', transform=args.rigid_augmentation_train, p=args.p, args=args)
    test_dataset = TeethSegmentationDataset(split='test', transform=args.rigid_augmentation_test, p=args.p, args=args)

    # Create DataLoader for both
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, test_loader
