import torch
import torch.nn as nn
import torch.optim as optim

# 간단한 신경망 정의
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(1, 1)  # 입력 1, 출력 1

    def forward(self, x):
        return self.fc(x)

# 데이터 생성
x = torch.tensor([[1.0], [2.0], [3.0]])
y = torch.tensor([[2.0], [4.0], [6.0]])

# 모델 초기화
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 학습 루프
for epoch in range(100):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

print("학습 완료")

# 모델의 가중치와 편향 출력
print("가중치:", model.fc.weight.data)
print("편향:", model.fc.bias.data)