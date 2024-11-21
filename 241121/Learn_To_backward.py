import torch

# 입력값 설정
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
w = torch.tensor(4.0, requires_grad=True)

# 순전파 계산
q = x + y
z = q * w

# 역전파 계산
z.backward()

# 기울기 출력
print("dz/dx:", x.grad)  # dz/dx = 4
print("dz/dy:", y.grad)  # dz/dy = 4
print("dz/dw:", w.grad)  # dz/dw = 5