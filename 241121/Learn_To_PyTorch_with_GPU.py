import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 데이터 생성 (간단한 이진 분류 문제)
torch.manual_seed(42)
X = torch.randn(1000, 2)  # 1000개의 샘플, 각 샘플은 2개의 특징
y = (X[:, 0]**2 + X[:, 1]**2 < 1).float().unsqueeze(1)  # 반지름 1인 원 내부는 1, 외부는 0

# 데이터를 GPU로 전송
X, y = X.to(device), y.to(device)

# 데이터셋 및 데이터로더 정의
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 신경망 모델 정의
class SimpleANN(nn.Module):
    def __init__(self):
        super(SimpleANN, self).__init__()
        self.layer1 = nn.Linear(2, 16)  # 입력층(2) -> 은닉층(16)
        self.layer2 = nn.Linear(16, 8)  # 은닉층(16) -> 은닉층(8)
        self.layer3 = nn.Linear(8, 1)   # 은닉층(8) -> 출력층(1)
        self.activation = nn.ReLU()    # ReLU 활성화 함수
        self.sigmoid = nn.Sigmoid()    # Sigmoid 활성화 함수 (출력층)

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x

# 모델 초기화 및 GPU로 전송
model = SimpleANN().to(device)

# 손실 함수 및 옵티마이저 정의
criterion = nn.BCELoss()  # 이진 분류를 위한 Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 학습 루프
num_epochs = 20
for epoch in range(num_epochs):
    for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
        # 순전파
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)

        # 역전파 및 가중치 업데이트
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 에포크마다 손실 출력
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 학습된 모델로 결과 확인
with torch.no_grad():
    X_test = torch.tensor([[0.5, 0.5], [-1.5, -1.5]]).to(device)  # 테스트 데이터
    predictions = model(X_test)
    print("\nTest Data Predictions:")
    for i, pred in enumerate(predictions):
        print(f"Data {X_test[i].tolist()} -> Prediction: {pred.item():.4f}")