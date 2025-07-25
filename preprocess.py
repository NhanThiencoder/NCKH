import numpy as np
import os
from eccodes import *

# Danh sách file GRIB
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
batch_index = 0
batch_size = 1000

for file_path in grib_files:
    with open(file_path, 'rb') as f:
        timestep_data = {}
        while True:
            gid = codes_grib_new_from_file(f)
            if gid is None:
                break
            try:
                varname = codes_get(gid, 'shortName')
                values = codes_get_values(gid)
                Ni = codes_get(gid, 'Ni')
                Nj = codes_get(gid, 'Nj')

                if varname in variables_needed:
                    arr_2d = np.array(values).reshape(Nj, Ni)
                    timestep_data[varname] = arr_2d

                if len(timestep_data) == len(variables_needed):
                    sample = np.stack([timestep_data[var] for var in variables_needed], axis=0).astype(np.float32)
                    batch.append(sample)
                    timestep_data = {}

                    if len(batch) == batch_size:
                        np.save(f"{output_dir}/batch_{batch_index}.npy", np.array(batch))
                        print(f"Saved batch_{batch_index}.npy with {len(batch)} samples")
                        batch = []
                        batch_index += 1

            except Exception as e:
                print(f"Error: {e}")
            finally:
                codes_release(gid)

if batch:
    np.save(f"{output_dir}/batch_{batch_index}.npy", np.array(batch))
    print(f"Saved final batch_{batch_index}.npy with {len(batch)} samples")