import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import StormCNNTransformer
from dataset import StormTrackDataset
from prepare_data import prepare_data

# --- Cấu hình ---
feature_path = 'all_features.npy'
label_path = 'storm_labels_grib_padded.npy'
sequence_length = 20
batch_size = 32
num_epochs = 100  # Tăng epoch
learning_rate = 1e-5  # Giảm learning rate
weight_decay = 1e-6  # Tăng regularization
patience = 10  # Tăng patience

# --- Tạo dataset và DataLoader ---
train_loader, val_loader, test_loader = prepare_data(
    feature_path=feature_path,
    label_path=label_path,
    sequence_length=sequence_length,
    batch_size=batch_size,
    split_ratio=(0.7, 0.15, 0.15),
    num_workers=0,
    pad=True
)

# --- Model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = StormCNNTransformer(cnn_out_dim=256, d_model=256, nhead=8, num_layers=6, dropout=0.1).to(device)  # Tăng kích thước

# --- Loss, Optimizer, và Scheduler ---
criterion = nn.MSELoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# --- Huấn luyện ---
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    train_losses = []
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        mask = ~torch.isnan(y).any(dim=-1)
        if torch.any(torch.isnan(y)):
            y = torch.nan_to_num(y, nan=0.0)
        output = model(x, mask)

        loss = criterion(output, y)
        mask = mask.unsqueeze(-1).expand_as(loss)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_losses.append(loss.item() if not np.isnan(loss.item()) else 0)

    model.eval()
    val_losses = []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            mask = ~torch.isnan(y).any(dim=-1)
            if torch.any(torch.isnan(y)):
                y = torch.nan_to_num(y, nan=0.0)
            output = model(x, mask)
            loss = criterion(output, y)
            mask = mask.unsqueeze(-1).expand_as(loss)
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
            val_losses.append(loss.item() if not np.isnan(loss.item()) else 0)

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)
    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step(val_loss)

    print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_storm_transformer.pth')
        print(f"New best val loss: {best_val_loss:.6f} - Saved model")
    else:
        patience_counter += 1
        print(f"No improvement for {patience_counter} epochs")

    if patience_counter >= patience:
        print(f"Early stopping triggered after {epoch + 1} epochs")
        break

print("\nTraining complete. Best val loss:", best_val_loss)

# --- Đánh giá trên test set ---
model.load_state_dict(torch.load('best_storm_transformer.pth'))
model.eval()
test_losses = []
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        mask = ~torch.isnan(y).any(dim=-1)
        if torch.any(torch.isnan(y)):
            y = torch.nan_to_num(y, nan=0.0)
        output = model(x, mask)
        loss = criterion(output, y)
        mask = mask.unsqueeze(-1).expand_as(loss)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        test_losses.append(loss.item() if not np.isnan(loss.item()) else 0)

test_loss = np.mean(test_losses)
print(f"Test Loss: {test_loss:.6f}")