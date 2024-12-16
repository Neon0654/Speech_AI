import os
import librosa
import numpy as np
from pathlib import Path

# Hàm trích xuất đặc trưng MFCC từ file âm thanh
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)  # Đọc file âm thanh
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Trích xuất MFCC
    mfcc = np.mean(mfcc.T, axis=0)  # Tính trung bình các MFCC theo thời gian
    return mfcc

# Đọc tất cả các file trong các thư mục con
folder_path = "C:/Users/tthoa/OneDrive/Desktop/recorgnition/DataRecorgnite"
features = []
labels = []

for sub_folder in os.listdir(folder_path):  # Duyệt qua các thư mục con
    sub_folder_path = os.path.join(folder_path, sub_folder)
    if os.path.isdir(sub_folder_path):  # Kiểm tra xem có phải thư mục không
        for file_name in os.listdir(sub_folder_path):  # Duyệt qua các file
            if file_name.endswith(".mp3"):  # Chỉ xử lý file .mp3
                file_path = os.path.join(sub_folder_path, file_name)
                mfcc = extract_features(file_path)  # Trích xuất MFCC
                features.append(mfcc)
                labels.append(sub_folder)  # Gán nhãn bằng tên thư mục

# Chuyển đổi list thành numpy array
X = np.array(features)
y = np.array(labels)
