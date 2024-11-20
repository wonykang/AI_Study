import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 데이터 생성
np.random.seed(42)
X = np.random.rand(20, 1) * 10  # 0~10 사이의 값
y = 2 * X + 1 + np.random.randn(20, 1) * 2  # y = 2x + 1 + 노이즈

# 학습용 및 테스트용 데이터 분리
X_train = X[:15]
y_train = y[:15]
X_test = X[15:]
y_test = y[15:]

# 다항 회귀 모델 (2차 & 15차)
poly2 = PolynomialFeatures(degree=2)
poly15 = PolynomialFeatures(degree=15)

X_train_poly2 = poly2.fit_transform(X_train)
X_train_poly15 = poly15.fit_transform(X_train)

X_test_poly2 = poly2.transform(X_test)
X_test_poly15 = poly15.transform(X_test)

# 선형 회귀 모델 학습
model2 = LinearRegression().fit(X_train_poly2, y_train)
model15 = LinearRegression().fit(X_train_poly15, y_train)

# 예측
y_pred2 = model2.predict(X_test_poly2)
y_pred15 = model15.predict(X_test_poly15)

# 시각화
plt.scatter(X, y, color='blue', label='Data')
plt.plot(np.sort(X_train, axis=0), model2.predict(poly2.transform(np.sort(X_train, axis=0))), color='green', label='Degree 2')
plt.plot(np.sort(X_train, axis=0), model15.predict(poly15.transform(np.sort(X_train, axis=0))), color='red', label='Degree 15')
plt.legend()
plt.show()

# 성능 평가
print("Degree 2 MSE:", mean_squared_error(y_test, y_pred2))
print("Degree 15 MSE:", mean_squared_error(y_test, y_pred15))