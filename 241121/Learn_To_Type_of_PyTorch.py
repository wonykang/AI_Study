import torch.nn as nn
import torch

#ReLU (Rectified Linear Unit)
relu = nn.ReLU()
# Leaky ReLU
leaky_relu = nn.LeakyReLU(negative_slope=0.01)
#  Parametric ReLU (PReLU)
prelu = nn.PReLU()
# ELU (Exponential Linear Unit)
elu = nn.ELU(alpha=1.0)
# Sigmoid
sigmoid = nn.Sigmoid()
# Tanh (Hyperbolic Tangent)
tanh = nn.Tanh()
#  Softmax
softmax = nn.Softmax(dim=1)
#  LogSoftmax
log_softmax = nn.LogSoftmax(dim=1)
# Swish
def swish(x):
    return x * torch.sigmoid(x)
# Softplus
softplus = nn.Softplus()

# PyTorch에서의 활성화 함수 사용 예

# 입력 데이터
x = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)

# 활성화 함수 적용
relu = nn.ReLU()
leaky_relu = nn.LeakyReLU()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()
softmax = nn.Softmax(dim=0)

print("ReLU:", relu(x))
print("Leaky ReLU:", leaky_relu(x))
print("Sigmoid:", sigmoid(x))
print("Tanh:", tanh(x))
print("Softmax:", softmax(x))