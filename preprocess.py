import numpy as np
import os
from eccodes import *

grib_files = sorted([
    'data/2000-2012_1h.grib',
    'data/2000-2012_7h.grib',
    'data/2000-2012_13h.grib',
    'data/2000-2012_19h.grib',
    'data/2013-2025_1h.grib',
    'data/2013-2025_7h.grib',
    'data/2013-2025_13h.grib',
    'data/2013-2025_19h.grib'
])

variables_needed = [
    "avg_cpr", "avg_lsprate", "avg_rorwe", "avg_ssurfror", "avg_surfror", "avg_tprate",
    "slhf", "ssr", "sshf", "tcc", "tclw", "e", "sro", "cp", "lsp", "tcrw", "vitoe",
    "10u", "10v", "2t", "msl", "sst", "tp", "mx2t", "mn2t"
]

output_dir = "processed_batches"
os.makedirs(output_dir, exist_ok=True)

batch = []
batch_times = []  # Lưu thời gian tương ứng
batch_index = 0
batch_size = 1000
grid_size = None

for file_path in grib_files:
    if not os.path.exists(file_path):
        print(f"Lỗi: File {file_path} không tồn tại")
        continue

    with open(file_path, 'rb') as f:
        while True:
            gid = codes_grib_new_from_file(f)
            if gid is None:
                break
            try:
                data_date = codes_get(gid, 'dataDate')
                data_time = codes_get(gid, 'dataTime')
                time_str = f"{data_date}{data_time:04d}"
                varname = codes_get(gid, 'shortName')
                values = codes_get_values(gid)
                Ni = codes_get(gid, 'Ni')
                Nj = codes_get(gid, 'Nj')

                if grid_size is None:
                    grid_size = (Nj, Ni)
                elif grid_size != (Nj, Ni):
                    print(f"Cảnh báo: Kích thước lưới không nhất quán tại {time_str}, biến {varname}")
                    continue

                if varname in variables_needed:
                    arr_2d = np.array(values).reshape(Nj, Ni)
                    if 'timestep_data' not in locals():
                        timestep_data = {}
                    timestep_data[varname] = arr_2d

                if len(timestep_data) == len(variables_needed):
                    sample = np.stack([timestep_data[var] for var in variables_needed], axis=0).astype(np.float32)
                    batch.append(sample)
                    batch_times.append(time_str)
                    timestep_data = {}

                    if len(batch) == batch_size:
                        np.savez(f"{output_dir}/batch_{batch_index}.npz", data=np.array(batch), times=np.array(batch_times))
                        print(f"Saved batch_{batch_index}.npz with {len(batch)} samples")
                        batch = []
                        batch_times = []
                        batch_index += 1

            except Exception as e:
                print(f"Lỗi khi xử lý thông điệp GRIB tại {file_path}: {e}")
            finally:
                codes_release(gid)

if batch:
    np.savez(f"{output_dir}/batch_{batch_index}.npz", data=np.array(batch), times=np.array(batch_times))
    print(f"Saved final batch_{batch_index}.npz with {len(batch)} samples")