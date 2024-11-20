import torch
x = torch.tensor([[1.0, 1.0]])
w = torch.tensor([[2.0], [2.0]])
b = torch.tensor([-3.0])

z = torch.matmul(x, w) + b
output = torch.sigmoid(z)
print(output)