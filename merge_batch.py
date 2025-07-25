import numpy as np
import glob
import re

def merge_batches(batch_dir="processed_batches", output_file="all_features.npy"):
    batch_paths = sorted(
        glob.glob(f"{batch_dir}/batch_*.npy"),
        key=lambda x: int(re.search(r"batch_(\d+)\.npy", x).group(1))
    )
    all_data = []

    for i, path in enumerate(batch_paths):
        print(f"[{i+1:>2}/{len(batch_paths)}] Nạp {path} ...")
        batch = np.load(path)
        all_data.append(batch)

    merged = np.concatenate(all_data, axis=0)
    np.save(output_file, merged)
    print(f"✅ Đã lưu '{output_file}' với shape: {merged.shape}")

if __name__ == "__main__":
    merge_batches()