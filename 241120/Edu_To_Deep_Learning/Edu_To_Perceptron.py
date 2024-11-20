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