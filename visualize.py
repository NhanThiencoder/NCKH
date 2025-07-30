import torch
import matplotlib.pyplot as plt
from model import StormCNNTransformer
from prepare_data import prepare_data
import numpy as np

# --- Cấu hình ---
feature_path = 'all_features.npy'
label_path = 'storm_labels_grib_padded.npy'
sequence_length = 20
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Tạo DataLoader ---
train_loader, val_loader, test_loader = prepare_data(
    feature_path=feature_path,
    label_path=label_path,
    sequence_length=sequence_length,
    batch_size=batch_size,
    split_ratio=(0.7, 0.15, 0.15),
    num_workers=0,
    pad=True
)

# --- Tải mô hình đã huấn luyện (đồng bộ với train.py) ---
model = StormCNNTransformer(cnn_out_dim=256, d_model=256, nhead=8, num_layers=6, dropout=0.1).to(device)
model.load_state_dict(torch.load('best_storm_transformer.pth'))
model.eval()

# --- Lấy dữ liệu và dự đoán ---
lat_scale = 180.0
lon_scale = 360.0

with torch.no_grad():
    for i, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        mask = ~torch.isnan(y).any(dim=-1)
        output = model(x, mask)

        if i == 0:
            y_actual = y.cpu().numpy() * [lat_scale, lon_scale]
            y_pred_raw = output.cpu().numpy()
            y_pred = y_pred_raw * [lat_scale, lon_scale]

            # Xử lý NaN và giới hạn
            y_actual = np.nan_to_num(y_actual, nan=0.0, posinf=90, neginf=-90)
            y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=90, neginf=-90)
            y_actual = np.clip(y_actual, [-90, -180], [90, 180])
            y_pred = np.clip(y_pred, [-90, -180], [90, 180])

            # Debug
            print(f"y_actual shape: {y_actual.shape}")
            print(f"y_pred shape: {y_pred.shape}")
            print(f"y_actual sample: {y_actual[0]}")
            print(f"y_pred_raw sample (before scaling): {y_pred_raw[0]}")
            print(f"y_pred sample: {y_pred[0]}")
            if np.any(np.isnan(y_actual)) or np.any(np.isnan(y_pred)):
                print("Warning: NaN detected")

            # Vẽ
            for j in range(min(3, batch_size)):
                plt.figure(figsize=(10, 6))
                plt.plot(y_actual[j, :, 0], y_actual[j, :, 1], 'b-', label='Actual Path')
                plt.plot(y_pred[j, :, 0], y_pred[j, :, 1], 'r--', label='Predicted Path')
                valid_steps = mask[j].cpu().numpy()
                plt.scatter(y_actual[j, valid_steps, 0], y_actual[j, valid_steps, 1], c='blue', s=10, label='Actual Points')
                plt.scatter(y_pred[j, valid_steps, 0], y_pred[j, valid_steps, 1], c='red', s=10, label='Predicted Points')

                plt.title(f'Storm Path - Sample {j+1}')
                plt.xlabel('Latitude (°)')
                plt.ylabel('Longitude (°)')
                plt.legend()
                plt.grid(True)
                plt.xlim(-90, 90)
                plt.ylim(-180, 180)
                plt.show()

            break

print("Biểu đồ đường đi đã được tạo!")