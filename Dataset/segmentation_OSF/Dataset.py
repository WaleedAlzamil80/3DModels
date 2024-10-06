import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
import numpy as np
import trimesh

from factories.sampling_factory import get_sampling_technique

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
                        obj_path = os.path.join(region_dir, sample_id, f'{sample_id}_{region}.obj')
                        label_path = os.path.join(region_dir, sample_id, f'{sample_id}_{region}.json')
                        data_list.append((obj_path, label_path))
        return data_list

    def __len__(self):
        return len(self.data_list)

    def _load_obj_file(self, obj_path):
        """Load .obj file and return vertices and faces."""
        mesh_data = trimesh.load(obj_path)
        # vertices = mesh_data.vertices.astype(np.float32)
        vertices = torch.tensor(mesh_data.vertices, dtype=torch.float32).unsqueeze(0)
        # faces = mesh_data.faces
        return self.sampling_fn(vertices, fea=None, args=self.args)

    def _load_labels(self, label_path):
        """Load labels from the JSON file."""
        with open(label_path, 'r') as f:
            file = json.load(f)
        labels = np.maximum(0, np.array(file['labels']) - 10 - 2 * ((np.array(file['labels']) // 10) - 1))
        return torch.tensor(labels, dtype=torch.long), torch.tensor(self.jaw_to_idx[file['jaw']], dtype=torch.long)

    def __getitem__(self, idx):
        obj_path, label_path = self.data_list[idx]

        centroids, vertices, fea_vertices, fe_labels, idx = self._load_obj_file(obj_path)
        labels, jaw = self._load_labels(label_path)

        return vertices.view(-1, 3), labels[idx].view(-1), jaw

# Usage of the dataset
def OSF_data_loaders(args):
    # Create training and testing datasets
    train_dataset = TeethSegmentationDataset(split='train', p=args.p, args=args)
    test_dataset = TeethSegmentationDataset(split='test', p=args.p, args=args)

    # Create DataLoader for both
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, test_loader