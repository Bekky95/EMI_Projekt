import numpy as np
import torch
from torch.utils.data import Dataset


class MemotionDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.text = data["text"]
        self.image = data["image"]
        self.label = data["label"]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.text[idx], dtype=torch.float32),
            torch.tensor(self.image[idx], dtype=torch.float32),
            torch.tensor(self.label[idx], dtype=torch.long),
        )