import torch

class Perceptron(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 1)  # 입력 2, 출력 1

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

torch.manual_seed(42)
model = Perceptron()
x = torch.tensor([[0.0, 1.0]])
output = model(x)
print(output)
criterion = torch.nn.BCELoss()
y_true = torch.tensor([[1.0]])
y_pred = torch.tensor([[0.9]])
loss = criterion(y_pred, y_true)
print(loss.item())
x = torch.tensor([[1.0, 1.0]])
w = torch.tensor([[2.0], [2.0]])
b = torch.tensor([-3.0])

z = torch.matmul(x, w) + b
output = torch.sigmoid(z)
print(z, output)
w = torch.tensor(1.0, requires_grad=True)
x = torch.tensor(1.0)
y = torch.tensor(2.0)
learning_rate = 0.1

# 손실 계산
y_pred = w * x
loss = (y_pred - y) ** 2
loss.backward()

# 가중치 업데이트
print(w.grad)
w = w - learning_rate * w.grad
print(w.item())