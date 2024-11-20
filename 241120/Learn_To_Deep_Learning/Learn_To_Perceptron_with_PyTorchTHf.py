import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import numpy as np

# 데이터 생성 (2개의 클래스로 분류)
torch.manual_seed(42)
X, y = make_moons(n_samples=500, noise=0.1, random_state=42)
X = torch.tensor(X, dtype=torch.float32)  # 입력 데이터
y = torch.tensor(y, dtype=torch.long)    # 레이블 (0 또는 1)

# 데이터 시각화
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm")
plt.title("Generated Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# 다층 퍼셉트론 모델 정의
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)  # 은닉층
        self.output_layer = nn.Linear(hidden_dim, output_dim) # 출력층
        self.activation = nn.ReLU()  # 활성화 함수

    def forward(self, x):
        x = self.activation(self.hidden_layer(x))  # 은닉층 + 활성화 함수
        x = self.output_layer(x)  # 출력층
        return x

# 모델 초기화
input_dim = 2       # 입력 특징 수
hidden_dim = 16     # 은닉층 뉴런 수
output_dim = 2      # 출력 클래스 수 (0과 1)
model = MLP(input_dim, hidden_dim, output_dim)

# 손실 함수 (Cross Entropy Loss)
criterion = nn.CrossEntropyLoss()

# 옵티마이저 (Adam)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 학습 설정
epochs = 1000
losses = []

for epoch in range(epochs):
    # 1. 순전파
    outputs = model(X)
    loss = criterion(outputs, y)
    losses.append(loss.item())

    # 2. 역전파
    optimizer.zero_grad()
    loss.backward()

    # 3. 파라미터 업데이트
    optimizer.step()

    # 100번마다 손실 출력
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        predictions = torch.argmax(model(grid), axis=1)
    Z = predictions.reshape(xx.shape).numpy()
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y.numpy(), cmap="coolwarm")
    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# 결정 경계 시각화
plot_decision_boundary(X, y, model)