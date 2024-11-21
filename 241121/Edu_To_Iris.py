import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# 데이터 로드
iris = load_iris()
X = iris.data
y = iris.target

# 데이터 전처리
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# DataLoader 생성
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=16, shuffle=False)

# 모델 정의
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(4, 16)  # 입력: 4 (특징), 출력: 16
        self.fc2 = nn.Linear(16, 3)  # 출력: 3 (클래스 3개)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU 활성화 함수 적용
        x = self.fc2(x)  # 최종 출력
        return x

# 모델, 손실 함수, 옵티마이저 정의
model = ANN()
criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss는 다중 클래스 분류 문제에 적합
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 옵티마이저 사용

# 학습 함수
def train(model, train_loader, criterion, optimizer, num_epochs=100):
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
train(model, train_loader, criterion, optimizer, num_epochs=100)
evaluate(model, test_loader)

# 모델 저장
torch.save(model.state_dict(), 'iris_ann.pth')
print("Model saved as 'iris_ann.pth'")

# 모델 로드
model_loaded = ANN()
model_loaded.load_state_dict(torch.load('iris_ann.pth'))
model_loaded.eval()  # 평가 모드로 전환
