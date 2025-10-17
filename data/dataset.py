import torch
from torch.utils.data import Dataset
import numpy as np

# Custom PyTorch Dataset for tabular data.
# Stores features X and targets y as float32 and provides
# standard __len__ and __getitem__ methods for DataLoader.
class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]