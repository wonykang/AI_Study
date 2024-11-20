import numpy as np

# 활성화 함수 (계단 함수)
def step_function(x):
    return np.where(x >= 0, 1, 0)

# 퍼셉트론 함수
def perceptron(x, w, b):
    z = np.dot(x, w) + b
    return step_function(z)

# 입력 데이터 (AND 게이트 예제)
x_data = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y_data = np.array([0, 0, 0, 1])  # AND 게이트의 출력

# 가중치와 편향 초기화
w = np.random.rand(2)
b = np.random.rand()

# 학습률
lr = 0.1

# 학습
for epoch in range(10):  # 10번 반복 학습
    for x, y in zip(x_data, y_data):
        y_pred = perceptron(x, w, b)
        error = y - y_pred
        w += lr * error * x  # 가중치 업데이트
        b += lr * error      # 편향 업데이트

# 결과 확인
for x in x_data:
    print(f"입력: {x}, 출력: {perceptron(x, w, b)}")