import numpy as np
import glob
import re


def merge_batches_incremental(batch_dir="processed_batches", output_file="all_features.npy"):
    batch_paths = sorted(
        glob.glob(f"{batch_dir}/batch_*.npy"),
        key=lambda x: int(re.search(r"batch_(\d+)\.npy", x).group(1)) if re.search(r"batch_(\d+)\.npy", x) else 0
    )
    if not batch_paths:
        raise FileNotFoundError(f"Không tìm thấy file .npy nào trong {batch_dir}")

    # Lấy shape từ batch đầu tiên
    first_batch = np.load(batch_paths[0], mmap_mode='r')
    sample_shape = first_batch.shape[1:]
    total_samples = sum(np.load(path, mmap_mode='r').shape[0] for path in batch_paths)

    # Tạo file memmap
    merged = np.memmap(output_file, dtype=np.float32, mode='w+', shape=(total_samples,) + sample_shape)

    offset = 0
    for i, path in enumerate(batch_paths):
        print(f"[{i + 1:>2}/{len(batch_paths)}] Nạp và gộp {path} ...")
        batch = np.load(path, mmap_mode='r')
        if batch.shape[1:] != sample_shape:
            print(f"Cảnh báo: Shape của {path} không khớp: {batch.shape[1:]} != {sample_shape}")
            continue
        merged[offset:offset + batch.shape[0]] = batch
        offset += batch.shape[0]

    print(f"Đã lưu '{output_file}' với shape: {merged.shape}")
    return merged