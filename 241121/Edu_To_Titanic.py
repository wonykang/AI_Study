import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 데이터 로드 및 전처리
data = pd.read_csv('titanic.csv')

# 결측치 처리: 나이와 승선지 결측치를 평균값으로 채움
data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# 범주형 변수 인코딩: 성별과 승선지 인코딩
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# 특성과 레이블 분리
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].values
y = data['Survived'].values

# 데이터 정규화
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 훈련/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 텐서로 변환
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 모델 정의
class TitanicANN(nn.Module):
    def __init__(self):
        super(TitanicANN, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 64)  # 입력층: 특성 개수 -> 64 뉴런
        self.fc2 = nn.Linear(64, 32)                # 은닉층: 64 -> 32 뉴런
        self.fc3 = nn.Linear(32, 1)                 # 출력층: 32 -> 1 뉴런 (생존 여부)
        self.sigmoid = nn.Sigmoid()                 # 시그모이드 활성화 함수

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU 활성화 함수
        x = torch.relu(self.fc2(x))  # ReLU 활성화 함수
        x = self.fc3(x)              # 최종 출력층
        x = self.sigmoid(x)          # 이진 분류이므로 시그모이드 함수로 확률값 출력
        return x

# 모델, 손실 함수, 옵티마이저 정의
model = TitanicANN()
criterion = nn.BCELoss()  # 이진 교차 엔트로피 손실 함수
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 함수
def train(model, X_train, y_train, criterion, optimizer, num_epochs=100):
    model.train()  # 훈련 모드로 설정
    for epoch in range(num_epochs):
        optimizer.zero_grad()  # 기울기 초기화

        # 순전파
        y_pred = model(X_train)

        # 손실 계산
        loss = criterion(y_pred, y_train)

        # 역전파
        loss.backward()

        # 파라미터 업데이트
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 평가 함수
def evaluate(model, X_test, y_test):
    model.eval()  # 평가 모드로 설정
    with torch.no_grad():  # 평가 중에는 기울기 계산을 하지 않음
        y_pred = model(X_test)
        predicted = (y_pred > 0.5).float()  # 확률을 0.5 이상이면 생존(1), 아니면 사망(0)으로 분류
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)  # 정확도 계산
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

# 학습 실행
train(model, X_train, y_train, criterion, optimizer, num_epochs=100)

# 평가 실행
evaluate(model, X_test, y_test)

# 모델 저장
torch.save(model.state_dict(), 'titanic_model.pth')
print("Model saved as 'titanic_model.pth'")

# 모델 로드
model_loaded = TitanicANN()
model_loaded.load_state_dict(torch.load('titanic_model.pth'))
model_loaded.eval()  # 평가 모드로 전환
