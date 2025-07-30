
import numpy as np
import os
print(os.path.exists('all_features.npy'))
data = np.load('storm_labels_grib_cleaned.npz')
labels = data['labels']
names = data['names']
print("Hình dạng nhãn:", labels.shape)
print("Danh sách tên bão:", names)
padded_data = np.load('storm_labels_grib_padded.npz')
padded_labels = padded_data['labels']
padded_names = padded_data['names']
print("Hình dạng nhãn đã padding:", padded_labels.shape)
print("Danh sách tên bão đã padding:", padded_names)