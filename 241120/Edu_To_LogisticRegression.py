from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split # Import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# 다중 클래스 데이터 생성
# Set n_redundant to 0 to ensure the sum is less than n_features
X, y = make_classification(n_samples=150, n_features=4, n_classes=3, n_informative=3, n_redundant=0, random_state=42)

# 데이터 분리 및 모델 학습
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
multi_log_model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
multi_log_model.fit(X_train, y_train)

# 정확도 계산
multi_accuracy = accuracy_score(y_test, multi_log_model.predict(X_test))
print("Multi-Class Accuracy:", multi_accuracy)