import torch
from torch.utils.data import Dataset, DataLoader
from preprocess.gaussianize import *
import pandas as pd

class dataloader(Dataset):
    def __init__(self, data, length):
        assert len(data) >= length
        self.data = data
        self.length = length

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx:idx+self.length]).reshape(-1, self.length).to(torch.float32)

    def __len__(self):
        return max(len(self.data)-self.length, 0)
    
