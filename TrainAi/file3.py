from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from file2 import X_train, X_test, y_train, y_test  # Import dữ liệu từ file2

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SVM phân loại
model = svm.SVC(kernel='linear')  # Sử dụng kernel tuyến tính
model.fit(X_train, y_train)

# Dự đoán và đánh giá
y_pred = model.predict(X_test)

