from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np


# 데이터 (이진 분류 문제로 변경)
X = np.array([[1], [2], [3], [4], [5]])  # 특성
y = np.array([0, 1, 0, 1, 0])  # 이진 클래스 값 (0 또는 1)
# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# L1 규제 적용
l1_model = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)
l1_model.fit(X_train, y_train)

# 예측 및 정확도
y_pred_l1 = l1_model.predict(X_test)
accuracy_l1 = accuracy_score(y_test, y_pred_l1)
print("L1 Regularized Accuracy:", accuracy_l1)