from torch.utils.data import Dataset, DataLoader
import os
import json
import numpy as np
import trimesh

class TeethSegmentationDataset(Dataset):
    def __init__(self, root_dir, split='train', test_ids_file='private-testing-set/private-testing-set.txt', transform=None, p=7):
        """
        Args:
            root_dir (string): Directory with all the parts (data_part_{1-6}).
            split (string): 'train' or 'test' to select the appropriate dataset.
            test_ids_file (string): Path to the txt file containing IDs for testing.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.p = p
        self.test_ids = self._load_test_ids(test_ids_file)
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
            part_dir = os.path.join(self.root_dir, f'data_part_{part}')
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
        vertices = mesh_data.vertices.astype(np.float32)
        # faces = mesh_data.faces
        return vertices

    def _load_labels(self, label_path):
        """Load labels from the JSON file."""
        with open(label_path, 'r') as f:
            labels = json.load(f)
        return np.maximum(0, np.array(labels['labels']) - 10 - 2 * ((np.array(labels['labels']) // 10) - 1))

    def __getitem__(self, idx):
        obj_path, label_path = self.data_list[idx]

        # Load data
        vertices = self._load_obj_file(obj_path)
        labels = self._load_labels(label_path)

        # sample = {'vertices': vertices, 'labels': labels}

        # if self.transform:
        #    sample = self.transform(sample)

        return vertices, labels

# Usage of the dataset
def get_data_loaders(args):
    # Create training and testing datasets
    train_dataset = TeethSegmentationDataset(root_dir=args.path, split='train', test_ids_file=args.test_ids, p=args.p)
    test_dataset = TeethSegmentationDataset(root_dir=args.path, split='test', test_ids_file=args.test_ids, p=args.p)

    # Create DataLoader for both
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, test_loader