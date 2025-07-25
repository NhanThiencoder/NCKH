import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def extract_storm_tracks(excel_path):
    df = pd.read_excel(excel_path, header=6)
    storms = []
    for _, row in df.iterrows():
        storm = {
            'year': row.iloc[0],
            'name': row.iloc[1],
            'start_time': row.iloc[2],
            'end_time': row.iloc[3],
            'track': []
        }
        i = 4
        while i < len(row) - 1:
            lon = row.iloc[i]
            lat = row.iloc[i + 1]
            if pd.notna(lon) and pd.notna(lat):
                time_offset = (i - 4) // 2 * 6
                storm['track'].append({
                    'lon': float(lon),
                    'lat': float(lat),
                    'time_offset': time_offset
                })
            else:
                break
            i += 2
        storms.append(storm)
    return storms

def create_labels(storms, batch_files, output_path=None):
    total_samples = 0
    for file in batch_files:
        batch = np.load(file)
        total_samples += batch.shape[0]

    labels = np.full((total_samples, 2), -999, dtype=np.float32)

    start_date = datetime(2000, 1, 1, 0, 0)
    time_step = timedelta(hours=6)
    grib_times = [start_date + i * time_step for i in range(total_samples)]

    for storm in storms:
        start_time = pd.to_datetime(storm['start_time'], dayfirst=True, errors='coerce')
        if pd.isna(start_time):
            print(f"Bỏ qua cơn bão {storm['name']} do lỗi ngày giờ: {storm['start_time']}")
            continue
        track = storm['track']
        for point in track:
            hours_from_start = point['time_offset']
            target_time = start_time + timedelta(hours=hours_from_start)
            for i, grib_time in enumerate(grib_times):
                if abs((grib_time - target_time).total_seconds() / 3600) <= 3:
                    labels[i] = [point['lat'], point['lon']]
                    break

    if output_path:
        np.save(output_path, labels)

    return labels

if __name__ == "__main__":
    excel_path = "LongandLat.xlsx"
    batch_files = sorted(glob.glob("processed_batches/batch_*.npy"))
    storms = extract_storm_tracks(excel_path)
    labels = create_labels(storms, batch_files, output_path="storm_labels_grib.npy")

    print(f"Đã trích xuất {len(storms)} cơn bão")
    print(f"Tổng số nhãn: {labels.shape[0]}")
    print("10 nhãn đầu tiên:")
    print(labels[:10])
