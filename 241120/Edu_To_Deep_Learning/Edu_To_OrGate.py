import numpy as np
x = np.array([0, 1])
w = np.array([1, 1])
b = (-0.5)  # 편향 값


def perceptron(x, w, b):
    return 1 if np.dot(x, w) + b > 0 else 0


print(perceptron(x, w, b))  #(계산: (0 \cdot 1 + 1 \cdot 1 - 0.5 = 0.5 > 0), 출력: 1)