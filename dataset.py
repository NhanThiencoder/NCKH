import numpy as np
import torch
from torch.utils.data import Dataset
import os

class StormTrackDataset(Dataset):
    def __init__(self, feature_path, label_path, sequence_length=20,
                 h_start=20, h_end=53, w_start=15, w_end=46,
                 lat_scale=180.0, lon_scale=360.0, pad=False):
        self.feature_path = feature_path
        self.label_path = label_path
        self.sequence_length = sequence_length
        self.h_start, self.h_end = h_start, h_end
        self.w_start, self.w_end = w_start, w_end
        self.lat_scale, self.lon_scale = lat_scale, lon_scale
        self.pad = pad

        # Sử dụng memmap để tải features
        self.features = np.memmap(self.feature_path, dtype='float32', mode='r',
                               shape=(28160, 25, 73, 61))
        self.labels = np.load(self.label_path)

        if self.features.shape[0] != self.labels.shape[0]:
            raise ValueError(f"Số mẫu không khớp: features={self.features.shape[0]}, labels={self.labels.shape[0]}")

        # Chuẩn hóa features (có thể cần tính min_val, max_val từng phần)
        self.min_val = np.min(self.features)  # Có thể chậm, cần tối ưu
        self.max_val = np.max(self.features)
        self.features = np.memmap(self.feature_path, dtype='float32', mode='r',
                               shape=(28160, 25, 73, 61))  # Tải lại để chuẩn hóa
        self.features = (self.features - self.min_val) / (self.max_val - self.min_val + 1e-8)

        # Tạo valid_indices
        valid_mask = np.all(self.labels != -999, axis=1)
        self.valid_indices = []
        for i in range(len(self.labels) - sequence_length + 1):
            segment_mask = valid_mask[i:i+sequence_length]
            valid_count = np.sum(segment_mask)
            if valid_count == sequence_length:
                self.valid_indices.append((i, sequence_length))
            elif self.pad and valid_count >= 15:
                self.valid_indices.append((i, valid_count))

        if not self.valid_indices:
            raise ValueError("Không tìm thấy đoạn chuỗi hợp lệ nào")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        try:
            start, valid_count = self.valid_indices[idx]
            end = start + self.sequence_length

            x = self.features[start:end].copy()
            y = self.labels[start:end].copy()

            if x.shape[0] < self.sequence_length:
                pad_len = self.sequence_length - x.shape[0]
                x = np.pad(x, ((0, pad_len), (0, 0), (0, 0), (0, 0)), mode='constant')
                y = np.pad(y, ((0, pad_len), (0, 0)), mode='constant', constant_values=-999)
            elif x.shape[0] > self.sequence_length:
                x = x[:self.sequence_length]
                y = y[:self.sequence_length]

            x = x[:, :, self.h_start:self.h_end, self.w_start:self.w_end]

            # Thay -999 bằng giá trị gần nhất
            valid_mask = y != -999
            if not np.any(valid_mask):
                raise ValueError(f"Không có nhãn hợp lệ tại chỉ số {start}")
            for j in range(y.shape[0]):
                if not valid_mask[j].all():
                    if j > 0 and valid_mask[j - 1].all():
                        y[j] = y[j - 1]
                    else:
                        y[j] = [0, 0]  # Thay NaN bằng [0, 0]

            # Kiểm tra và chuẩn hóa
            if np.any(y > [90, 180]) or np.any(y < [-90, -180]):
                print(f"Cảnh báo: Nhãn vượt giới hạn tại chỉ số {start}: {y}")
            y_norm = y / np.array([self.lat_scale, self.lon_scale])
            if np.any(np.isnan(y_norm)) or np.any(y_norm > 0.5) or np.any(y_norm < -0.5):
                print(f"Cảnh báo: Giá trị chuẩn hóa chứa NaN hoặc vượt [-0.5, 0.5] tại {start}: {y_norm}")
                y_norm = np.nan_to_num(y_norm, nan=0.0, posinf=0.5, neginf=-0.5)  # Thay NaN bằng 0

            latlon = y_norm[:, :, np.newaxis, np.newaxis]
            latlon = np.tile(latlon, (1, 1, x.shape[2], x.shape[3]))
            x = np.concatenate([x, latlon], axis=1)

            return torch.tensor(x, dtype=torch.float32), torch.tensor(y_norm, dtype=torch.float32)
        except Exception as e:
            raise RuntimeError(f"Lỗi khi lấy mẫu {idx}: {e}") 

if __name__ == "__main__":
    try:
        dataset = StormTrackDataset(
            feature_path="all_features.npy",
            label_path="storm_labels_grib_padded.npy",
            sequence_length=20,
            pad=True
        )
        print(f"Số đoạn hợp lệ: {len(dataset)}")
        x, y = dataset[0]
        print(f"Shape features: {x.shape}")
        print(f"Shape labels: {y.shape}")
        lengths = [count for _, count in dataset.valid_indices]
        print(f"Phân bố độ dài đoạn: {np.bincount(lengths)}")
    except Exception as e:
        print(f"Lỗi khi kiểm tra dataset: {e}")