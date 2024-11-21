import torch
import torch.nn as nn
import torch.optim as optim

# 데이터 준비 (x1, x2 -> y)
x_data = torch.tensor([[0.5, 0.3], [0.7, 0.8], [0.2, 0.4]], dtype=torch.float32)  # 입력 데이터
y_data = torch.tensor([[1.0], [0.0], [1.0]], dtype=torch.float32)  # 레이블 (정답)

# 모델 클래스 정의
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # 입력층 -> 은닉층 (2 뉴런)
        self.hidden = nn.Linear(2, 2)  # 2 입력 -> 2 은닉층
        # 은닉층 -> 출력층 (1 뉴런)
        self.output = nn.Linear(2, 1)  # 2 은닉층 -> 1 출력층
        self.activation = nn.Sigmoid()  # Sigmoid 활성화 함수

    def forward(self, x):
        h = self.activation(self.hidden(x))  # 은닉층 활성화
        y_hat = self.output(h)  # 출력층 활성화
        return y_hat

# 모델 생성
model = SimpleNN()

# 손실 함수 및 옵티마이저 정의
criterion = nn.MSELoss()  # 손실 함수: Mean Squared Error
optimizer = optim.SGD(model.parameters(), lr=0.1)  # SGD 옵티마이저

# 학습 루프
num_epochs = 1000  # 학습 반복 횟수
for epoch in range(num_epochs):
    # 순전파: 예측값 계산
    y_hat = model(x_data)

    # 손실 계산
    loss = criterion(y_hat, y_data)

    # 역전파: 기울기 계산 및 업데이트
    optimizer.zero_grad()  # 기울기 초기화
    loss.backward()  # 역전파
    optimizer.step()  # 가중치 업데이트

    # 100 에포크마다 결과 출력
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 학습 후 결과 확인
print("\n학습 결과:")
with torch.no_grad():  # 학습 모드 중지
    y_hat = model(x_data)
    print("예측값:", y_hat.squeeze().numpy())
    print("실제값:", y_data.squeeze().numpy())