import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.spatial import cKDTree
import os

# Đường dẫn file Excel và thư mục batch .npz
excel_file = "LongandLat.xlsx"
batch_dir = "processed_batches/"

# Tải tất cả file .npz từ batch_0 đến batch_28
batch_files = [f"{batch_dir}batch_{i}.npz" for i in range(29)]
all_data = []
all_times = np.unique(np.concatenate([np.load(f"{batch_dir}batch_{i}.npz")['times'] for i in range(29)]))

for file in batch_files:
    with np.load(file) as data:
        all_data.append(data['data'])

all_data = np.concatenate(all_data)
print("Các thời gian trong all_times sau khi ghép:", all_times)
print("Hình dạng của all_data:", all_data.shape)

# Định nghĩa lưới kinh độ và vĩ độ
n_lons, n_lats = 73, 61
lon_min, lon_max = 102, 117
lat_min, lat_max = 6, 24
lons = np.linspace(lon_min, lon_max, n_lons)
lats = np.linspace(lat_min, lat_max, n_lats)
print("Kinh độ:", lons[:5], "...", lons[-5:])
print("Vĩ độ:", lats[:5], "...", lats[-5:])

# Tạo cây KDTree
grid_points = np.array([[lon, lat] for lon in lons for lat in lats])
tree = cKDTree(grid_points)

# Hàm chuyển đổi thời gian
def parse_time(time_str):
    if isinstance(time_str, datetime):
        return time_str
    try:
        return datetime.strptime(time_str, '%d-%m-%Y %H:%M')
    except ValueError as e:
        print(f"Lỗi định dạng thời gian {time_str}: {e}")
        return None

# Hàm tạo nhãn
def create_labels(storm_data, start_time, time_offset=6):
    storm_labels = np.zeros((len(storm_data), 2))
    for j in range(len(storm_data)):
        target_time = start_time + timedelta(hours=j * time_offset)
        time_str = target_time.strftime('%Y%m%d%H%M')
        idx = np.where(all_times == time_str)[0]
        if len(idx) > 0:
            grid_data = all_data[idx[0]]
            long, lat = storm_data[j]
            dist, idx_grid = tree.query([long, lat])
            grid_idx = np.unravel_index(idx_grid, (n_lons, n_lats))
            mean_value = np.mean(grid_data[:, grid_idx[0], grid_idx[1]])
            storm_labels[j] = [long, lat]
        else:
            print(f"Cảnh báo: Không tìm thấy thời gian {time_str} cho bão tại bước {j}")
            storm_labels[j] = [-999, -999]
    return storm_labels

# Đọc file Excel
df = pd.read_excel(excel_file)
columns = df.columns.tolist()
print("Các cột trong file Excel:", columns)

# Xử lý từng bão
storm_labels_all = []
storm_names = []
start_col = 'Start'
max_steps = 20
for i in range(len(df)):
    storm_start = df[start_col].iloc[i]
    start_time = parse_time(storm_start)
    if start_time is None:
        continue
    path = []
    for j in range(max_steps):
        long_col = f'Long.{j}' if f'Long.{j}' in columns else f'Long .{j}'
        lat_col = f'Lat.{j}' if f'Lat.{j}' in columns else f'Lat .{j}'
        if long_col in columns and lat_col in columns:
            long = pd.to_numeric(df[long_col].iloc[i], errors='coerce')
            lat = pd.to_numeric(df[lat_col].iloc[i], errors='coerce')
            if not np.isnan(long) and not np.isnan(lat):
                path.append([long, lat])
    if path:
        storm_labels = create_labels(path, start_time)
        storm_labels_all.append(storm_labels)
        storm_name = df['Name'].iloc[i] if 'Name' in columns else f"Storm_{i}"
        storm_names.append(storm_name)
        print(f"Đã trích xuất nhãn cho bão {storm_name}")

# Chuẩn hóa kích thước bằng padding
if storm_labels_all:
    max_length = max(len(labels) for labels in storm_labels_all)
    padded_labels = [np.pad(labels, ((0, max_length - len(labels)), (0, 0)), mode='constant', constant_values=-999) for labels in storm_labels_all]
    storm_labels_all = np.array(padded_labels)
else:
    storm_labels_all = np.array([])

# Làm sạch nhãn
if storm_labels_all.size > 0:
    storm_labels_all[storm_labels_all == -999] = np.nan
    valid_mask = ~np.isnan(storm_labels_all).all(axis=(1, 2))  # Kiểm tra từng mẫu
    valid_mask = valid_mask.any(axis=1)  # Đảm bảo mask đúng với từng mẫu
    cleaned_labels = storm_labels_all[valid_mask]
    cleaned_names = [storm_names[i] for i in range(len(storm_names)) if valid_mask[i]]
else:
    cleaned_labels = np.array([])
    cleaned_names = []

# Lưu file .npz với nhãn đã làm sạch và names
np.savez('storm_labels_grib_cleaned.npz', labels=cleaned_labels, names=cleaned_names)
print(f"Đã lưu {len(cleaned_labels)} hàng hợp lệ vào storm_labels_grib_cleaned.npz")

# Tính min/max
if cleaned_labels.size > 0:
    lat_values = cleaned_labels[:, :, 1]  # Gán trục 1 là vĩ độ
    lon_values = cleaned_labels[:, :, 0]  # Gán trục 0 là kinh độ
    valid_lat_mask = (~np.isnan(lat_values)) & (lat_values >= 6) & (lat_values <= 24)
    valid_lon_mask = (~np.isnan(lon_values)) & (lon_values >= 102) & (lon_values <= 117)

    if np.any(valid_lat_mask):
        min_lat = np.nanmin(lat_values[valid_lat_mask])
        max_lat = np.nanmax(lat_values[valid_lat_mask])
    else:
        min_lat = np.nan
        max_lat = np.nan
        print("Cảnh báo: Không có giá trị vĩ độ hợp lệ để tính min/max.")

    if np.any(valid_lon_mask):
        min_lon = np.nanmin(lon_values[valid_lon_mask])
        max_lon = np.nanmax(lon_values[valid_lon_mask])
    else:
        min_lon = np.nan
        max_lon = np.nan
        print("Cảnh báo: Không có giá trị kinh độ hợp lệ để tính min/max.")
else:
    min_lat = np.nan
    max_lat = np.nan
    min_lon = np.nan
    max_lon = np.nan
    print("Cảnh báo: Không có dữ liệu hợp lệ để tính min/max.")

print(f"Min Latitude: {min_lat}, Max Latitude: {max_lat}")
print(f"Min Longitude: {min_lon}, Max Longitude: {max_lon}")

# Tải và kiểm tra all_features.npy
features = np.load('all_features.npy') if os.path.exists('all_features.npy') else None
if features is not None:
    if features.shape[0] > cleaned_labels.shape[0]:
        padding = np.full((features.shape[0] - cleaned_labels.shape[0], cleaned_labels.shape[1], 2), -999, dtype=cleaned_labels.dtype)
        padded_labels = np.vstack((cleaned_labels, padding))
        padded_names = np.array(cleaned_names + ['Padded_' + str(i) for i in range(features.shape[0] - cleaned_labels.shape[0])])
    else:
        padded_labels = cleaned_labels
        padded_names = cleaned_names
else:
    # Padding cố định nếu không có features
    padding_size = 10
    padding = np.full((padding_size, cleaned_labels.shape[1] if cleaned_labels.size > 0 else max_steps, 2), -999, dtype=cleaned_labels.dtype if cleaned_labels.size > 0 else float)
    padded_labels = np.vstack((cleaned_labels, padding)) if cleaned_labels.size > 0 else padding
    padded_names = np.array(cleaned_names + ['Padded_' + str(i) for i in range(padding_size)])

# Lưu file padded
np.savez('storm_labels_grib_padded.npz', labels=padded_labels, names=padded_names)
print(f"Đã tạo storm_labels_grib_padded.npz với {padded_labels.shape[0]} mẫu")