import torch
from torch.utils.data import DataLoader, random_split
from dataset import StormTrackDataset
import numpy as np
import os

def prepare_data(
    feature_path="all_features.npy",
    label_path="storm_labels_grib_padded.npy",
    sequence_length=20,
    batch_size=16,
    split_ratio=(0.7, 0.15, 0.15),
    num_workers=0,
    pad=True
):
    """
    Chuẩn bị dataset và DataLoader cho train/val/test.
    """
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"File {feature_path} không tồn tại")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"File {label_path} không tồn tại")

    try:
        dataset = StormTrackDataset(
            feature_path=feature_path,
            label_path=label_path,
            sequence_length=sequence_length,
            pad=pad
        )
    except Exception as e:
        raise RuntimeError(f"Lỗi khi khởi tạo dataset: {e}")

    total_len = len(dataset)
    if total_len == 0:
        raise ValueError("Dataset rỗng, không thể chia train/val/test")
    print(f"Số đoạn hợp lệ: {total_len}")

    train_len = int(split_ratio[0] * total_len)
    val_len = int(split_ratio[1] * total_len)
    test_len = total_len - train_len - val_len

    if batch_size > min(train_len, val_len, test_len):
        raise ValueError(f"Batch size ({batch_size}) lớn hơn số mẫu: train={train_len}, val={val_len}, test={test_len}")

    try:
        train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])
    except Exception as e:
        raise RuntimeError(f"Lỗi khi chia dataset: {e}")

    try:
        np.save("train_indices.npy", train_set.indices)
        np.save("val_indices.npy", val_set.indices)
        np.save("test_indices.npy", test_set.indices)
        print(f"Đã lưu chỉ số train/val/test vào file .npy")
    except Exception as e:
        print(f"Cảnh báo: Không thể lưu chỉ số chia dataset: {e}")

    try:
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False
        )
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False
        )
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False
        )
    except Exception as e:
        raise RuntimeError(f"Lỗi khi tạo DataLoader: {e}")

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Cấu hình
    feature_path = "all_features.npy"
    label_path = "storm_labels_grib.npy"
    sequence_length = 20
    batch_size = 16
    split_ratio = (0.7, 0.15, 0.15)
    num_workers = 0
    pad = True  # Định nghĩa pad trong scope chính

    # Chuẩn bị data
    try:
        train_loader, val_loader, test_loader = prepare_data(
            feature_path='all_features.npy',
            label_path='storm_labels_grib_padded.npy',  # Sử dụng file đã padding
            sequence_length=sequence_length,
            batch_size=batch_size,
            split_ratio=(0.7, 0.15, 0.15),
            num_workers=0,
            pad=True
        )

        # Kiểm tra các loader
        for split, loader in [("Train", train_loader), ("Val", val_loader), ("Test", test_loader)]:
            print(f"\nKiểm tra {split} loader:")
            try:
                for inputs, targets in loader:
                    print(f"Input shape: {inputs.shape}")
                    print(f"Target shape: {targets.shape}")
                    break
            except Exception as e:
                print(f"Lỗi khi tải batch từ {split} loader: {e}")

        # Kiểm tra phân bố độ dài đoạn
        if pad:
            lengths = [count for _, count in train_loader.dataset.dataset.valid_indices]
            print(f"Phân bố độ dài đoạn: {np.bincount(lengths)}")

    except Exception as e:
        print(f"Lỗi khi chuẩn bị data: {e}")