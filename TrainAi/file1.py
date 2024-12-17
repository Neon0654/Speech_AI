import os
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler

def extract_features(file_path, sr=22050, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs, axis=1)

def load_data(data_dir, batch_size=32):

    X, y = [], []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for file_name in os.listdir(label_dir):
                file_path = os.path.join(label_dir, file_name)
                if file_name.endswith(".mp3"):
                    features = extract_features(file_path)
                    X.append(features)
                    y.append(label)

    X = np.array(X)
    y = np.array(y)

    return X, y

# Data loading and feature extraction (call load_data)
data_dir = "C:/Users/tthoa/OneDrive/Desktop/recorgnition/DataRecorgnite"
X, y = load_data(data_dir)
