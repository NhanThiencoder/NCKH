import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def extract_storm_tracks(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['time'])
    storms = []

    grouped = df.groupby(['year', 'name'])

    for (year, name), group in grouped:
        group = group.sort_values('time')
        track = []
        for _, row in group.iterrows():
            track.append({
                'time': row['time'],
                'lat': float(row['lat']),
                'lon': float(row['lon']),
            })

        storms.append({
            'year': year,
            'name': name,
            'track': track
        })

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
        for point in storm['track']:
            target_time = point['time']
            for i, grib_time in enumerate(grib_times):
                if abs((grib_time - target_time).total_seconds() / 3600) <= 3:
                    labels[i] = [point['lat'], point['lon']]
                    break

    if output_path:
        np.save(output_path, labels)

    return labels

if __name__ == "__main__":
    csv_path = "LongandLat.xlsx"
    batch_files = sorted(glob.glob("processed_batches/batch_*.npy"))
    storms = extract_storm_tracks(csv_path)
    labels = create_labels(storms, batch_files, output_path="storm_labels_grib.npy")

    print(f"Đã trích xuất {len(storms)} cơn bão")
    print(f"Tổng số nhãn: {labels.shape[0]}")
    print("10 nhãn đầu tiên:")
    print(labels[:10])