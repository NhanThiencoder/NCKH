import numpy as np
import torch
from torch.utils.data import Dataset


class StormTrackDataset(Dataset):
    def __init__(self, feature_path, label_path, sequence_length=20):
        self.sequence_length = sequence_length

        self.features = np.load(feature_path)  # [T, 25, 73, 61]
        self.labels = np.load(label_path)      # [T, 2]

        assert len(self.features) == len(self.labels), "Số lượng feature và label không khớp!"

    def __len__(self):
        return len(self.features) - self.sequence_length + 1

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.sequence_length]  # [T, 25, 73, 61]
        y = self.labels[idx:idx + self.sequence_length]    # [T, 2]

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        return x_tensor, y_tensor