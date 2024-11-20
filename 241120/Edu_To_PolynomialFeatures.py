from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import numpy as np

# 데이터
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1.5, 4.5, 9.1, 16.2, 25.3])

# 2차 다항 회귀
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_model.fit(X, y)

# 예측
y_pred_poly = poly_model.predict(X)
print("Polynomial Predictions:", y_pred_poly)