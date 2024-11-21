import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# 간단한 모델 정의
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 1)
)

# 옵티마이저 정의
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 학습률 스케줄러 정의
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# 학습 루프
for epoch in range(50):  # 50 에포크 동안 학습
    # ... (데이터 로딩 및 순전파/역전파 생략)
    optimizer.step()  # 가중치 업데이트
    scheduler.step()  # 학습률 스케줄 업데이트

    print(f"Epoch {epoch+1}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")