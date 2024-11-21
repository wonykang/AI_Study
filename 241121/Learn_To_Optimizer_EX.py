import torch
import torch.nn as nn
import torch.optim as optim

# 간단한 모델 정의
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 1)
)

# 손실 함수와 옵티마이저 정의
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 임의 데이터
x = torch.randn(64, 10)
y = torch.randn(64, 1)

# 학습 루프
for epoch in range(100):
    optimizer.zero_grad()  # 기울기 초기화
    output = model(x)  # 순전파
    loss = criterion(output, y)  # 손실 계산
    loss.backward()  # 역전파
    optimizer.step()  # 가중치 업데이트

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")