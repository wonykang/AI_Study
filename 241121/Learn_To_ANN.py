import torch
import torch.nn as nn
from torchsummary import summary

# ANN 모델 정의
class ANNModel(nn.Module):
    def __init__(self):
        super(ANNModel, self).__init__()
        self.hidden_layer = nn.Linear(3, 2)  # 입력층(3) → 은닉층(2)
        self.output_layer = nn.Linear(2, 1)  # 은닉층(2) → 출력층(1)
        self.activation_hidden = nn.ReLU()  # 은닉층 활성화 함수
        self.activation_output = nn.Sigmoid()  # 출력층 활성화 함수

    def forward(self, x):
        x = self.activation_hidden(self.hidden_layer(x))  # 은닉층 연산
        x = self.activation_output(self.output_layer(x))  # 출력층 연산
        return x

# 모델 초기화
model = ANNModel()

# 모델 구조 요약 출력
summary(model, input_size=(3,))

# 가중치 확인
for name, param in model.named_parameters():
    print(f"{name}:")
    print(param.data)
    print()