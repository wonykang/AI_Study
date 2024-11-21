import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 커스텀 손실 함수 정의 (MSE + L2 규제)
class CustomLoss(nn.Module):
    def __init__(self, alpha=0.01):
        super(CustomLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()  # CrossEntropyLoss로 변경
        self.alpha = alpha  # L2 규제의 가중치

    def forward(self, y_pred, y_true, model):
        # CrossEntropyLoss는 (batch_size, num_classes) 형식의 예측과 (batch_size,) 형식의 정수형 레이블을 받습니다.
        loss = self.cross_entropy(y_pred, y_true)  # CrossEntropyLoss 계산
        l2_reg = sum(torch.sum(param**2) for param in model.parameters())  # L2 규제 계산
        return loss + self.alpha * l2_reg  # MSE 손실 + L2 규제

# 모델 정의
class MultiClassClassifier(nn.Module):
    def __init__(self):
        super(MultiClassClassifier, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # 입력: 28*28 (MNIST 이미지 크기)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # 출력: 10 (MNIST는 10개의 클래스)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28*28)  # 이미지를 1차원 벡터로 변환 (28x28 크기)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 데이터 로드 (정규화 포함)
transform = transforms.Compose([
    transforms.ToTensor(),  # 이미지를 텐서로 변환
    transforms.Normalize((0.5,), (0.5,))  # 평균 0, 표준편차 1로 정규화
])

# 훈련 데이터
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 검증 데이터
val_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# 모델, 커스텀 손실 함수 정의
model = MultiClassClassifier()
criterion = CustomLoss(alpha=0.01)  # 커스텀 손실 함수 사용

# 옵티마이저 (SGD 사용)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 학습 루프
num_epochs = 5
for epoch in range(num_epochs):
    model.train()  # 모델을 훈련 모드로 설정
    for X_batch, y_batch in train_loader:
        # 순전파
        y_pred = model(X_batch)
        
        # 커스텀 손실 계산
        loss = criterion(y_pred, y_batch, model)
        
        # 기울기 초기화 및 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # 검증 데이터로 성능 평가
    model.eval()  # 모델을 평가 모드로 설정
    with torch.no_grad():  # 검증 시에는 기울기를 계산할 필요 없음
        total = 0
        correct = 0
        for X_batch, y_batch in val_loader:
            y_pred = model(X_batch)
            predictions = torch.argmax(y_pred, dim=1)  # 가장 높은 확률의 클래스를 예측
            total += y_batch.size(0)
            correct += (predictions == y_batch).sum().item()  # 예측값과 실제값 비교
        val_accuracy = 100 * correct / total
        print(f"Validation Accuracy: {val_accuracy:.2f}%")

# 모델 저장
torch.save(model.state_dict(), 'model.pth')
print("Model saved as 'model.pth'")

# 저장된 모델 로드
model_loaded = MultiClassClassifier()
model_loaded.load_state_dict(torch.load('model.pth'))
print("Model loaded from 'model.pth'")

# 모델을 평가 모드로 전환
model_loaded.eval()

# 예시로 첫 번째 배치의 이미지를 사용하여 예측 수행
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        outputs = model_loaded(X_batch)
        _, predicted = torch.max(outputs, 1)
        print(f"Predicted labels: {predicted[:5]}")  # 첫 5개 예측값 출력
        break