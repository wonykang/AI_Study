import numpy as np
import matplotlib.pyplot as plt
w_val = []
cost_val = []

n_sample = 200
x = np.random.randn(n_sample)
y = 2 * x + 4 + np.random.randn(n_sample)
# plt.scatter(x, y)

n_epoch = 10  # 반복횟수
lr = 0.2      # 학습속도

w = np.random.uniform()
b = np.random.uniform()

for epoch in range(n_epoch):
  y_pred = w * x + b
  cost = np.abs(y_pred - y).mean()  # MAE
  w_diff = ((y_pred - y) * x).mean()
  b_diff = (y_pred - y).mean()
  print(f'{epoch:2} w={w:.6f}, b={b:.6f}, cost={cost:.6f}, w_diff={w_diff:.6f}, b_diff={b_diff:.6f}')

  w = w - lr * w_diff
  b = b - lr * b_diff
  # plt.plot(x, y_pred)

  w_val.append(w)
  cost_val.append(cost)

plt.plot(range(n_epoch), cost_val)
plt.show()