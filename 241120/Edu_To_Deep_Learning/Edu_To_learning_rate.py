import torch

w = torch.tensor(1.0, requires_grad=True)
x = torch.tensor(1.0)
y = torch.tensor(2.0)
learning_rate = 0.1

# 손실 계산
y_pred = w * x
loss = (y_pred - y) ** 2
loss.backward()

# 가중치 업데이트
w = w - learning_rate * w.grad
print(w.item()) #(손실의 그래디언트: (2(w \cdot x - y) \cdot x = 2(1 - 2) \cdot 1 = -2), 업데이트: (1 - 0.1 \cdot (-2) = 0.8))