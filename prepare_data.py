import torch
from torch.utils.data import DataLoader, random_split
from dataset import StormTrackDataset

# ==== Cấu hình ====
sequence_length = 20
batch_size = 16
feature_path = "all_features.npy"         # [T, 25, 73, 61]
label_path = "storm_labels_grib.npy"      # [T, 2]
split_ratio = (0.7, 0.15, 0.15)

# ==== Load Dataset ====
dataset = StormTrackDataset(
    feature_path=feature_path,
    label_path=label_path,
    sequence_length=sequence_length
)

# ==== Chia train/val/test ====
total_len = len(dataset)
train_len = int(split_ratio[0] * total_len)
val_len = int(split_ratio[1] * total_len)
test_len = total_len - train_len - val_len

train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

# ==== Dataloader ====
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# ==== Kiểm tra thử ====
if __name__ == "__main__":
    for inputs, targets in train_loader:
        print("Input shape:", inputs.shape)   # [B, T, 25, 73, 61]
        print("Target shape:", targets.shape) # [B, T, 2]
        break