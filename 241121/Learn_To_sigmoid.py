import torch
import torch.nn as nn

# 데이터 생성
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# 활성화 함수 적용
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()
relu = nn.ReLU()
softmax = nn.Softmax(dim=0)

print("Sigmoid:", sigmoid(x))
print("Tanh:", tanh(x))
print("ReLU:", relu(x))
print("Softmax:", softmax(x))