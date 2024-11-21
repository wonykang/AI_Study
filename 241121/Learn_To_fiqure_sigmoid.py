import numpy as np
import matplotlib.pyplot as plt

# 입력 데이터 범위
x = np.linspace(-10, 10, 100)

# 활성화 함수 정의
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # 안정성을 위한 정규화
    return exp_x / exp_x.sum()

# 각 활성화 함수 계산
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)
y_softmax = softmax(x)  # Softmax는 벡터로 작동, 전체 합이 1

# 그래프 그리기
plt.figure(figsize=(12, 8))

# Sigmoid
plt.subplot(2, 2, 1)
plt.plot(x, y_sigmoid, label="Sigmoid")
plt.title("Sigmoid Function")
plt.grid()
plt.legend()

# Tanh
plt.subplot(2, 2, 2)
plt.plot(x, y_tanh, label="Tanh", color="orange")
plt.title("Tanh Function")
plt.grid()
plt.legend()

# ReLU
plt.subplot(2, 2, 3)
plt.plot(x, y_relu, label="ReLU", color="green")
plt.title("ReLU Function")
plt.grid()
plt.legend()

# Softmax (Normalized for a single input)
plt.subplot(2, 2, 4)
plt.plot(x, y_softmax, label="Softmax (normalized)", color="red")
plt.title("Softmax Function")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()