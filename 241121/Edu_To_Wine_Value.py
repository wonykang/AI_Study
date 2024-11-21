import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 데이터 로드
data = pd.read_csv('winequality-red.csv', delimiter=';')
X = data.drop('quality', axis=1).values
y = data['quality'].values

# 데이터 전처리
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 텐서로 변환
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # 회귀 문제이므로 y는 2D 텐서여야 합니다
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 모델 정의
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 64)  # 첫 번째 레이어: 입력 -> 64 유닛
        self.fc2 = nn.Linear(64, 32)                # 두 번째 레이어: 64 -> 32 유닛
        self.fc3 = nn.Linear(32, 1)                 # 출력층: 32 -> 1 (회귀 문제이므로 출력은 하나의 실수 값)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU 활성화 함수
        x = torch.relu(self.fc2(x))  # ReLU 활성화 함수
        x = self.fc3(x)  # 최종 출력 (실수 값)
        return x

# 모델, 손실 함수, 옵티마이저 정의
model = ANN()
criterion = nn.MSELoss()  # 회귀 문제에서 사용되는 MSE 손실 함수
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 옵티마이저 사용

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
        loss = criterion(y_pred, y_test)  # MSE 손실 계산
        print(f"Test Loss: {loss.item():.4f}")

# 학습 실행
train(model, X_train, y_train, criterion, optimizer, num_epochs=100)

# 평가 실행
evaluate(model, X_test, y_test)

# 모델 저장
torch.save(model.state_dict(), 'wine_quality_model.pth')
print("Model saved as 'wine_quality_model.pth'")

# 모델 로드
model_loaded = ANN()
model_loaded.load_state_dict(torch.load('wine_quality_model.pth'))
model_loaded.eval()  # 평가 모드로 전환
