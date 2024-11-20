from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# 예시 데이터셋 (이진 분류)
# 여기서는 Iris 데이터셋을 사용하지만, 이진 분류만 사용할 것입니다.
data = load_iris()
X = data.data
y = data.target

# 이진 분류만 하려면, 클래스 0과 1만 선택 (클래스 2는 제외)
X = X[y != 2]
y = y[y != 2]

# 데이터를 훈련 세트와 테스트 세트로 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 훈련 (랜덤 포레스트)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 예측 값 구하기
y_pred = model.predict(X_test)

# 혼동 행렬 계산
conf_matrix = confusion_matrix(y_test, y_pred)

# 결과 출력
print("Confusion Matrix:\n", conf_matrix)

# 혼동 행렬 계산
# conf_matrix = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:\n", conf_matrix)