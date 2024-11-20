from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score

# 데이터
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2.1, 4.1, 6.2, 8.0, 9.9])

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 예측
predictions = model.predict(X_test)
print("Predictions:", predictions)


# R² 점수 계산
r2 = r2_score(y_test, predictions)
print("R² Score:", r2)