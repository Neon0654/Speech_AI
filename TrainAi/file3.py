import sounddevice as sd
import numpy as np
import librosa
import joblib
import wave

# Hàm ghi âm và lưu file âm thanh
def record_audio(filename, duration=5, samplerate=22050):
    print("Đang ghi âm...")
    # Ghi âm âm thanh từ microphone
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()  # Đợi cho đến khi ghi âm hoàn tất

    # Lưu âm thanh vào file WAV
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 byte cho mỗi mẫu âm thanh
        wf.setframerate(samplerate)
        wf.writeframes(audio.tobytes())

    print(f"Ghi âm xong, lưu tại {filename}")

# Hàm trích xuất đặc trưng MFCC từ file âm thanh
def extract_features(file_path, sr=22050, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs, axis=1)

# Hàm nhận diện giọng nói
def recognize_speech(model, filename):
    features = extract_features(filename)
    prediction = model.predict([features])
    return prediction[0]

# Ghi âm từ microphone và lưu vào file
audio_filename = "recorded_audio.wav"
record_audio(audio_filename, duration=5)  # Ghi âm 5 giây

# Tải mô hình đã huấn luyện
model = joblib.load('speech_recognition_model.pkl')

# Nhận diện giọng nói
predicted_label = recognize_speech(model, audio_filename)
print(f"Nhận diện giọng nói: {predicted_label}")
