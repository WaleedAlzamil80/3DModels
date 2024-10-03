import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from factories.sampling_factory import get_sampling_technique

class ModelNetDataset(Dataset):
    def __init__(self, args, split="train", transform=None):
        """
        Args:
            root_dir (string): Directory with all the 3D model data (train or test folder).
            split (string): Either "train" or "test".
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = args.path
        self.classes = os.listdir(self.root_dir)
        self.sampling_fn = get_sampling_technique(args.sampling)
        self.transform = transform
        self.files = []
        self.args =  args

        # Create a list of all (class, file_path) pairs
        for cls in self.classes:
            cls_folder = os.path.join(self.root_dir, cls, split)
            for file_name in os.listdir(cls_folder):
                if file_name.endswith('.off'):
                    file_path = os.path.join(cls_folder, file_name)
                    self.files.append((cls, file_path))

        # Map class names to indices
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        cls, file_path = self.files[idx]
        vertices, faces = read_off(file_path)

        vertices = torch.tensor(vertices, dtype=torch.float32)
        faces = torch.tensor(faces, dtype=torch.long)
        label = torch.tensor(self.class_to_idx[cls], dtype=torch.long)

        if self.transform:
            vertices = self.transform(vertices)
        return self.sampling_fn(vertices, self.args), label


def read_off(file_path):
    """Reads an .off file and returns the vertices and faces"""
    with open(file_path, 'r') as f:
        lines = f.readlines()

        # The first line should be "OFF"
        if lines[0].strip() != "OFF":
            raise ValueError("Not a valid OFF file")

        # Get the number of vertices and faces
        n_verts, n_faces, _ = map(int, lines[1].strip().split())

        # Load vertices
        vertices = np.array([list(map(float, line.strip().split())) for line in lines[2:2 + n_verts]])

        # Load faces
        faces = np.array([list(map(int, line.strip().split()[1:])) for line in lines[2 + n_verts:2 + n_verts + n_faces]])

        return vertices, faces

def modelnet_data_loaders(args):
    # Create training and testing datasets
    train_dataset = ModelNetDataset(split='train', args = args)
    test_dataset = ModelNetDataset(split='test', args = args)

    # Create DataLoader for both
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, test_loader