def create_labels(storms, batch_files, output_path=None):
    storm_paths = normalize_storm_paths(storms)
    num_storms = len(storm_paths)
    labels = np.full((num_storms, storm_paths.shape[1], 2), -999, dtype=np.float32)

    all_data = []
    all_times = []
    for file in batch_files:
        with np.load(file) as data:
            all_data.append(data['data'])
            all_times.append(data['times'])
    all_data = np.concatenate(all_data)
    all_times = np.concatenate(all_times)
    print("Các thời gian trong all_times:", all_times)  # Thêm dòng này

    for i, storm in enumerate(storms):
        start_time = storm['start_time']
        path = storm_paths[i]
        storm_labels = np.full((20, 2), -999, dtype=np.float32)
        for j, point in enumerate(storm['track']):
            if j >= 20:
                break
            target_time = start_time + timedelta(hours=point['time_offset'])
            time_str = target_time.strftime('%Y%m%d%H%M')
            idx = np.where(all_times == time_str)[0]
            if len(idx) > 0:
                storm_labels[j] = path[j]
            else:
                print(f"Cảnh báo: Không tìm thấy thời gian {time_str} cho bão {storm['name']} tại bước {j}")
        labels[i] = storm_labels

    if output_path:
        np.save(output_path, labels)

    return labels