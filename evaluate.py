import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import StormTrackDataset
from model import StormTransformer

# ==== Cấu hình ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
sequence_length = 20
label_path = "storm_labels_grib.npy"
grib_folder = "D:/NCKH/processed_batches/"
model_path = "storm_transformer_final.pth"  # hoặc

# ==== Tải dữ liệu ====
dataset = StormTrackDataset(label_path, grib_folder, sequence_length=sequence_length)
total_len = len(dataset)
train_len = int(0.7 * total_len)
val_len = int(0.15 * total_len)
test_len = total_len - train_len - val_len

_, _, test_dataset = torch.utils.data.random_split(dataset, [train_len, val_len, test_len])
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# ==== Mô hình ====
model = StormTransformer(input_dim=25, output_dim=2).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

criterion = nn.MSELoss()
total_loss = 0.0

# ==== Đánh giá ====
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()

average_loss = total_loss / len(test_loader)
print(f"[EVALUATE] Test Loss (MSE): {average_loss:.4f}")