import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from dataset import StormTrackDataset
from model import StormTransformer  # Bạn phải có model.py chứa lớp StormTransformer

# ==== Cấu hình ====
feature_path = "all_features.npy"
label_path = "storm_labels_grib.npy"
model_path = "best_storm_transformer.pth"  # File model đã lưu sau khi train
sequence_length = 20
batch_size = 1  # Đánh giá từng chuỗi một
min_val = -4002700.0
max_val = 2924932864.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Load dataset ====
dataset = StormTrackDataset(feature_path, label_path, sequence_length, min_val, max_val)
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# ==== Load mô hình ====
model = StormTransformer(input_dim=25, output_dim=2).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ==== Đánh giá ====
mse_list = []
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if inputs.ndim == 5:
            inputs = inputs.to(device)  # [1, T, 25, 73, 61]
            targets = targets.to(device)  # [1, T, 2]

            # Giản lược không gian: lấy trung bình theo (H, W)
            inputs = inputs.mean(dim=[-2, -1])  # [1, T, 25]

            outputs = model(inputs)  # [1, T, 2]

            mse = torch.mean((outputs - targets) ** 2).item()
            mse_list.append(mse)

            # Vẽ thử 1 vài dự đoán
            if batch_idx < 5:
                pred = outputs.squeeze(0).cpu().numpy()
                true = targets.squeeze(0).cpu().numpy()

                plt.figure()
                plt.plot(true[:, 1], true[:, 0], label='Thực tế', marker='o')     # Lon vs Lat
                plt.plot(pred[:, 1], pred[:, 0], label='Dự đoán', marker='x')
                plt.xlabel("Longitude")
                plt.ylabel("Latitude")
                plt.title(f"Storm Track Prediction - Sample {batch_idx}")
                plt.legend()
                plt.grid()
                plt.tight_layout()
                plt.show()

print(f"Test MSE: {np.mean(mse_list):.6f}")