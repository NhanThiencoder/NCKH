import numpy as np
import torch
from torch.utils.data import Dataset

import numpy as np
import torch
from torch.utils.data import Dataset

class StormTrackDataset(Dataset):
    def __init__(self, feature_path, label_path, sequence_length=20, min_val=None, max_val=None):
        self.sequence_length = sequence_length
        self.features = np.load(feature_path, mmap_mode='r')  # mmap: tránh load toàn bộ vào RAM
        self.labels = np.load(label_path)

        assert len(self.features) == len(self.labels), "Số lượng feature và label không khớp!"

        self.min_val = self.features.min() if min_val is None else min_val
        self.max_val = self.features.max() if max_val is None else max_val

    def __len__(self):
        return len(self.features) - self.sequence_length + 1

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.sequence_length]  # [T, 25, 73, 61]
        y = self.labels[idx:idx + self.sequence_length]    # [T, 2]

        # CHUẨN HÓA Ở ĐÂY (theo sample)
        x = (x - self.min_val) / (self.max_val - self.min_val + 1e-8)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)