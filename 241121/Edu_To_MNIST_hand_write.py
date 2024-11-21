import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 데이터셋 로드
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 모델 정의
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# 모델, 손실 함수, 옵티마이저 정의
model = ANN()
criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss는 분류 문제에 적합
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 옵티마이저 사용

# 훈련 함수
def train(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()  # 훈련 모드로 설정
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            # 순전파
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # 손실 계산

            # 기울기 초기화
            optimizer.zero_grad()
            
            # 역전파
            loss.backward()
            
            # 파라미터 업데이트
            optimizer.step()

            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

# 평가 함수
def evaluate(model, test_loader):
    model.eval()  # 평가 모드로 설정
    correct = 0
    total = 0
    with torch.no_grad():  # 평가 중에는 기울기를 계산할 필요 없음
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # 가장 큰 확률을 가진 클래스를 예측
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

# 학습 및 평가 실행
train(model, train_loader, criterion, optimizer, num_epochs=5)
evaluate(model, test_loader)