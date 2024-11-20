import numpy as np
import torch
# Perceptron 모델 정의
class Perceptron(torch.nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.fc = torch.nn.Linear(2, 1)  # 2개의 입력 (xx, yy)과 1개의 출력

    def forward(self, x):
        return self.fc(x)

# 모델 인스턴스 생성
model = Perceptron()

# 모델을 평가 모드로 설정 (Dropout, BatchNorm 비활성화)
model.eval()

xx, yy = np.meshgrid(np.arange(-1, 2, 0.01), np.arange(-1, 2, 0.01))
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

with torch.no_grad():
    predictions = model(grid)
Z = predictions.reshape(xx.shape).numpy()