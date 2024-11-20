# from sklearn.metrics import mean_squared_error

# # Assuming 'model' is your trained model and 'X_test' is your test data
# predictions = multi_log_model.predict(X_test) # Make predictions on the entire test set

# # RMSE 계산
# rmse = np.sqrt(mean_squared_error(y_test, predictions))
# print("RMSE:", rmse)

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

# 예시 데이터 생성 (임의의 회귀 문제 데이터)
X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)

# 데이터를 훈련 세트와 테스트 세트로 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 선형 회귀 모델 훈련
model = LinearRegression()
model.fit(X_train, y_train)

# 예측 값 계산
predictions = model.predict(X_test)

# RMSE 계산
rmse = np.sqrt(mean_squared_error(y_test, predictions))

# 결과 출력
print("RMSE:", rmse)
