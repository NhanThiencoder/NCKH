import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import StormTrackDataset
from model import StormTransformer

# ==== CẤU HÌNH ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_path = "all_features.npy"
label_path = "storm_labels_grib.npy"
sequence_length = 20
batch_size = 16
num_epochs = 20
learning_rate = 1e-4

# ==== LOAD DATA ====
full_dataset = StormTrackDataset(feature_path=feature_path, sequence_length=sequence_length)

total_len = len(full_dataset)
train_len = int(0.7 * total_len)
val_len = int(0.15 * total_len)
test_len = total_len - train_len - val_len

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_len, val_len, test_len])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# ==== KHỞI TẠO MÔ HÌNH ====
model = StormTransformer(input_dim=25, output_dim=2).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ==== HUẤN LUYỆN ====
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        # Nếu input là [B, T, 25, H, W] thì cần trung bình không gian
        if inputs.ndim == 5:
            inputs = inputs.mean(dim=[-2, -1])  # [B, T, 25]

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)  # [B, T, 2]
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # ==== ĐÁNH GIÁ ====
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            if inputs.ndim == 5:
                inputs = inputs.mean(dim=[-2, -1])
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    avg_train = train_loss / len(train_loader)
    avg_val = val_loss / len(val_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {avg_train:.6f} - Val Loss: {avg_val:.6f}")

    # ==== LƯU MODEL TỐT NHẤT ====
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), "best_storm_transformer.pth")
        print("==> Đã lưu mô hình tốt nhất!")

# ==== LƯU MÔ HÌNH CUỐI ====
torch.save(model.state_dict(), "storm_transformer_final.pth")
print("Đã lưu mô hình cuối cùng.")