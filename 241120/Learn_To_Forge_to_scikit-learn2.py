import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. 데이터 로드
data = load_breast_cancer()
X = data.data
y = data.target

# 2. 데이터 전처리: 표준화
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3. 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
data.data[0], X[0]
# 4. 로지스틱 회귀 모델 생성 및 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. 테스트 데이터로 예측
y_pred = model.predict(X_test)

# 6. 성능 평가
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

plt.plot(model.coef_.T, 'o')
plt.xticks(range(30), data.feature_names, rotation=90)
plt.grid()
# plt.show()

for C, marker in zip( [100, 1, 0.01], ['^', 'o', 'v']):
  model = LogisticRegression(C=C, max_iter=100000)
  model.fit(X_train, y_train)
  print( model.score(X_train, y_train), model.score(X_test, y_test) )
  plt.plot(model.coef_.T, marker, label=f'{C:.3f}')

plt.xticks(range(30), data.feature_names, rotation=90)
plt.legend()
plt.grid()
plt.show()