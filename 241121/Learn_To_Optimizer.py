import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 데이터 생성 (y = 2x + 1 + noise)
torch.manual_seed(42)  # 재현성을 위해 시드 고정
X = torch.linspace(0, 10, 100).unsqueeze(1)  # 입력 데이터
y = 2 * X + 1 + torch.randn(100, 1) * 2  # 출력 데이터 (노이즈 포함)

# 모델 정의 (단순 선형 모델)
model = nn.Linear(1, 1)  # 입력 1개, 출력 1개

# 손실 함수와 옵티마이저 정의
criterion = nn.MSELoss()  # 평균 제곱 오차 손실 함수
optimizer = optim.Adam(model.parameters(), lr=0.1)  # Adam 옵티마이저

# 학습 설정
epochs = 100
losses = []

for epoch in range(epochs):
    # 1. 순전파
    predictions = model(X)  # 모델 예측
    loss = criterion(predictions, y)  # 손실 계산

    # 2. 역전파
    optimizer.zero_grad()  # 기울기 초기화
    loss.backward()  # 역전파로 기울기 계산
    optimizer.step()  # 옵티마이저로 가중치 업데이트

    # 손실 기록
    losses.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# 학습 결과 확인
print("\n학습된 가중치와 절편:")
print(f"Weights: {model.weight.data.item()}, Bias: {model.bias.data.item()}")

# 결과 시각화
# 학습된 모델의 예측값
with torch.no_grad():
    y_pred = model(X)

# 데이터와 학습된 모델의 예측값 비교
plt.scatter(X.numpy(), y.numpy(), label="Original Data", alpha=0.6)
plt.plot(X.numpy(), y_pred.numpy(), color="red", label="Fitted Line")
plt.title("Linear Regression with Adam Optimizer")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()