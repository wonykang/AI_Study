import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np

# 데이터 생성 (2개의 클래스로 분류)
torch.manual_seed(42)
X, y = make_blobs(n_samples=200, centers=2, random_state=42)
X = torch.tensor(X, dtype=torch.float32)  # 입력 데이터
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # 레이블 (이진 분류, 0 또는 1)

# 데이터 시각화
plt.scatter(X[:, 0], X[:, 1], c=y.squeeze().numpy(), cmap="coolwarm")
plt.title("Generated Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# 단층 퍼셉트론 모델 정의
class SingleLayerPerceptron(nn.Module):
    def __init__(self, input_dim):
        super(SingleLayerPerceptron, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # 입력 크기: 2, 출력 크기: 1
        self.activation = nn.Sigmoid()  # 활성화 함수: 시그모이드

    def forward(self, x):
        return self.activation(self.linear(x))

# 모델 초기화
model = SingleLayerPerceptron(input_dim=2)

# 손실 함수 (Binary Cross Entropy Loss)
criterion = nn.BCELoss()

# 옵티마이저 (SGD)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 학습 설정
epochs = 1000
losses = []

for epoch in range(epochs):
    # 1. 순전파
    y_pred = model(X)

    # 2. 손실 계산
    loss = criterion(y_pred, y)
    losses.append(loss.item())

    # 3. 역전파
    optimizer.zero_grad()
    loss.backward()

    # 4. 파라미터 업데이트
    optimizer.step()

    # 100 에포크마다 손실 출력
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    
    # 결정 경계 시각화 함수
def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        predictions = model(grid)
    Z = predictions.reshape(xx.shape).numpy()
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.3, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y.squeeze().numpy(), cmap="coolwarm")
    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# 결정 경계 시각화
plot_decision_boundary(X, y, model)