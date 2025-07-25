import numpy as np
import torch
from torch.utils.data import Dataset


class StormTrackDataset(Dataset):
    def __init__(self, feature_path, sequence_length=20):
        self.sequence_length = sequence_length
        self.inputs = []
        self.labels = []

        data = np.load(feature_path)  # [T, 27] = 25 features + 2 labels
        total_seq = len(data) - sequence_length + 1

        for i in range(total_seq):
            seq = data[i:i + sequence_length]
            x = seq[:, :25]   # [T, 25] - 25 features
            y = seq[:, 25:]   # [T, 2]  - 2 tọa độ (lat, lon)
            self.inputs.append(torch.tensor(x, dtype=torch.float32))
            self.labels.append(torch.tensor(y, dtype=torch.float32))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]