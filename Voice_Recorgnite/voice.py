import librosa
import numpy as np
import sys
import os
sys.path.append(os.path.abspath("C:/Users/tthoa/OneDrive/Desktop/recorgnition/TrainAi"))  # Đảm bảo thêm đúng đường dẫn tuyệt đối

# Kiểm tra lại sys.path để xem đã thêm đúng chưa
print(sys.path)

from TrainAi.file1 import extract_features
from TrainAi.file3 import model, scaler


# Hàm ghi âm
import sounddevice as sd
from scipy.io.wavfile import write

output_path = "C:/Users/tthoa/OneDrive/Desktop/recorgnition/DataRecorgnite/Recorgnited"

def record_audio(output_path, duration=5, sr=16000):
    print("Đang ghi âm...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='int16')
    sd.wait()  # Chờ ghi âm hoàn tất
    write(output_path, sr, audio)  # Lưu file âm thanh
    print(f"Ghi âm hoàn tất. File được lưu tại: {output_path}")

# Ghi âm giọng nói
record_audio("recorded_audio.wav", duration=5)

# Trích xuất đặc trưng từ file ghi âm
features = extract_features("recorded_audio.wav")
print("Đặc trưng MFCC của file ghi âm:", features)

# Chuẩn hóa đặc trưng
features_scaled = scaler.transform([features])  # Chuẩn hóa

# Dự đoán với mô hình SVM
prediction = model.predict(features_scaled)
print("Kết quả dự đoán:", prediction[0])
