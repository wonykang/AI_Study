from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# 예시 데이터 생성 (임의의 회귀 문제 데이터)
X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)

# 데이터를 훈련 세트와 테스트 세트로 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 선형 회귀 모델 훈련
model = LinearRegression()
model.fit(X_train, y_train)

# 모델 계수와 절편 출력
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
