import numpy as np

x = np.array([0, 1])
w = np.array([1, 1])
b = (-0.5)  # 편향 값

def perceptron(x, w, b):
    return 1 if np.dot(x, w) + b > 0 else 0

def xor_gate(x):
    s1 = perceptron(x, [1, 1], -0.5)  # OR 게이트
    s2 = perceptron(x, [-1, -1], 1.5) # NAND 게이트
    return perceptron([s1, s2], [1, 1], -1.5)  # AND 게이트

print(xor_gate([1, 1]))