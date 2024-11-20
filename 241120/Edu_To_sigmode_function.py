import numpy as np

# 시그모이드 함수
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.array([0, 2, -3])
print("Sigmoid Results:", sigmoid(z))