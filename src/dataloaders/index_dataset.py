import torch

class IndexedDataset(torch.utils.data.Dataset):
    """
    Auxiliary dataset wrapper to also access the indices of the images.
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __getitem__(self, index):
        data = self.base_dataset[index]
        return torch.tensor(index), *data

    def __len__(self):
        return len(self.base_dataset)