from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
from file1 import X, y  # X và y đã được định nghĩa trong file1.py
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình SVM
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Kiểm tra độ chính xác
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Tạo pipeline để thực hiện chuẩn hóa và phân loại
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])

# Định nghĩa các siêu tham số cần thử nghiệm
param_grid = {
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['linear', 'rbf']
}

# Thực hiện grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Lấy mô hình tốt nhất
best_model = grid_search.best_estimator_

# Lưu mô hình đã huấn luyện
joblib.dump(best_model, 'speech_recognition_model.pkl')