import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

x = np.array([0, 1, 2, 3])
y = np.array([0, 1, 2, 3])

# 가중치(계수) 및 편향(절편)
w = 1
b = 0
y_pred1 = w * x + b
plt.plot(x, y_pred1, 'b-o')

w = 0.5
b = 0
y_pred2 = w * x + b
plt.plot(x, y_pred2, 'r-o')

plt.grid()
plt.show()

# def MSE(y_pred, y):
#   cost = np.sum((y_pred - y)**2) / len(y)
#   return cost

# MSE(y_pred1, y), MSE(y_pred2, y)