import torch
from torch.utils.data import Dataset

class IdxDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return idx, self.dataset[idx]
    
    def get_labels(self) -> torch.Tensor:
        return torch.as_tensor(list([self.dataset[i][1][0] for i in range(len(self))]))


    def get_group_labels(self) -> torch.Tensor:
        return torch.as_tensor(list([self.dataset[i][1][1] for i in range(len(self))]))