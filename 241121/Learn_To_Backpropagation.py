import torch
import torch.nn as nn
import torch.optim as optim

# 데이터 (x1, x2 -> y)
x = torch.tensor([[0.5, 0.3]], requires_grad=True)  # 입력
y = torch.tensor([[1.0]])  # 실제 값

# 가중치 초기화
W1 = torch.tensor([[0.1, 0.2], [0.3, 0.4]], requires_grad=True)  # 입력 -> 은닉층
b1 = torch.tensor([0.1, 0.1], requires_grad=True)
W2 = torch.tensor([[0.5], [0.6]], requires_grad=True)  # 은닉층 -> 출력층
b2 = torch.tensor([0.2], requires_grad=True)

# 활성화 함수
sigmoid = nn.Sigmoid()

# 순전파
h = sigmoid(x @ W1 + b1)  # 은닉층 출력
y_hat = sigmoid(h @ W2 + b2)  # 최종 출력

# 손실 계산
loss = 0.5 * (y_hat - y).pow(2).sum()  # MSE Loss

# 역전파
loss.backward()

# 결과 출력
print("Loss:", loss.item())
print("Gradient W1:", W1.grad)
print("Gradient W2:", W2.grad)