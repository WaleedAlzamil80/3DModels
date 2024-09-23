import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ModelNet10Dataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (string): Directory with all the 3D model data (train or test folder).
            split (string): Either "train" or "test".
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.classes = os.listdir(self.root_dir)
        self.transform = transform
        self.files = []

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


        sample = {
            'vertices': torch.tensor(vertices, dtype=torch.float32),
            'faces': torch.tensor(faces, dtype=torch.long),
            'label': torch.tensor(self.class_to_idx[cls], dtype=torch.long)
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


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

def get_dataloader():
    pass