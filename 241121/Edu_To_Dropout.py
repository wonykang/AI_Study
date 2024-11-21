import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# MLP 모델 정의 (드롭아웃 포함)
class MLPWithDropout(nn.Module):
    def __init__(self):
        super(MLPWithDropout, self).__init__()
        self.fc1 = nn.Linear(4, 16)  # 입력: 4, 출력: 16
        self.fc2 = nn.Linear(16, 8)  # 입력: 16, 출력: 8
        self.fc3 = nn.Linear(8, 2)   # 입력: 8, 출력: 2 (이진 분류)
        self.relu = nn.ReLU()        # ReLU 활성화 함수
        self.dropout = nn.Dropout(0.5)  # 50% 드롭아웃

    def forward(self, x):
        x = self.relu(self.fc1(x))  # 첫 번째 층
        x = self.dropout(self.relu(self.fc2(x)))  # 두 번째 층 + 드롭아웃
        x = self.fc3(x)  # 세 번째 층
        return x

# 데이터 생성 (예시로 임의의 데이터 사용)
# 100개의 샘플, 각 샘플은 4개의 특성 값
X = torch.randn(100, 4)
y = torch.randint(0, 2, (100,))  # 0 또는 1의 값을 가지는 타겟

# TensorDataset과 DataLoader로 데이터셋을 구성
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# 모델, 손실 함수, 최적화 알고리즘 정의
model = MLPWithDropout()
criterion = nn.CrossEntropyLoss()  # 이진 분류이므로 CrossEntropyLoss 사용
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
num_epochs = 20
for epoch in range(num_epochs):
    for X_batch, y_batch in train_loader:
        # 순전파
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        
        # 기울기 초기화 및 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 학습 상태 출력
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
