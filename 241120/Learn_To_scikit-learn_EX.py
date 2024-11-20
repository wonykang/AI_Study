# # 필요한 라이브러리 import
# import numpy as np
# import pandas as pd
# from sklearn.datasets import fetch_california_housing
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score

# # 1. 데이터셋 로드
# data = fetch_california_housing()
# X = data.data  # 특성(feature)
# y = data.target  # 타겟(target)

# # 데이터셋을 Pandas DataFrame으로 변환 (선택)
# df = pd.DataFrame(X, columns=data.feature_names)
# df['Target'] = y
# print(df.head())

# # 2. 학습용 및 테스트용 데이터 분리
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 3. 모델 생성 및 학습
# model = LinearRegression()
# model.fit(X_train, y_train)

# # 4. 예측
# y_pred = model.predict(X_test)

# # 5. 성능 평가
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"Mean Squared Error: {mse:.4f}")
# print(f"R^2 Score: {r2:.4f}")

# # 6. 회귀 계수 출력
# print("회귀 계수 (Weight):", model.coef_)
# print("절편 (Bias):", model.intercept_)

# 필요한 라이브러리 import
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. 데이터셋 로드
data = fetch_california_housing()
X = data.data  # 특성(feature)
y = data.target  # 타겟(target)

# 데이터셋을 Pandas DataFrame으로 변환 (선택)
df = pd.DataFrame(X, columns=data.feature_names)
df['Target'] = y
print(df.head())
df.info()
# 2. 학습용 및 테스트용 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 모델 생성 및 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 4. 예측
y_pred = model.predict(X_test)

# 5. 성능 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# 6. 회귀 계수 출력
print("회귀 계수 (Weight):", model.coef_)
print("절편 (Bias):", model.intercept_)
