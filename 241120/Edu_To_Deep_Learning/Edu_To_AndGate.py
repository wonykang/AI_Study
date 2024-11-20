import numpy as np

def perceptron(x, w, b):
    return 1 if np.dot(x, w) + b > 0 else 0

# AND 게이트
x = np.array([1, 1])  # 입력
w = np.array([1, 1])  # 가중치
b = -1.5              # 편향

print(perceptron(x, w, b))