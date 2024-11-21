import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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

# 데이터 로드
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 모델, 손실 함수, 최적화 정의
model = MultiClassClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프 (간단한 예시)
num_epochs = 5
for epoch in range(num_epochs):
    for X_batch, y_batch in train_loader:
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 모델 저장
torch.save(model.state_dict(), 'model.pth')
print("Model saved as 'model.pth'")

# 새로운 모델 인스턴스 생성 후 저장된 모델 로드
model_loaded = MultiClassClassifier()
model_loaded.load_state_dict(torch.load('model.pth'))
print("Model loaded from 'model.pth'")

# 모델을 평가 모드로 전환
model_loaded.eval()

# 예시로 첫 번째 배치의 이미지를 사용하여 예측 수행
with torch.no_grad():
    for X_batch, y_batch in train_loader:
        outputs = model_loaded(X_batch)
        _, predicted = torch.max(outputs, 1)
        print(f"Predicted labels: {predicted[:5]}")  # 첫 5개 예측값 출력
        break
