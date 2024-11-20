import matplotlib.pyplot as plt
# 오차가 최저가 되는 직선
import numpy as np
# 기울기 a를 최소제곱법으로 구하는 함수
def compute_a(x, y, mean_x, mean_y):
  # 분자부분
  dc = 0
  for i in range(len(x)):
    dc += (x[i] - mean_x) * (y[i] - mean_y)

  # 분모부분
  divisor = 0
  for i in range(len(x)):
    divisor += (x[i] - mean_x)**2

  a = dc / divisor
  return a

x = [8, 6, 4, 2]
y = [97, 91, 93, 81]
mx = np.mean(x)
my = np.mean(y)
a = compute_a(x, y, mx, my)  # 기울기
b = my - (mx * a)            # 절편

y_pred = [ a * x1 + b for x1 in x ]

plt.plot(x, y_pred, 'r-o')
plt.plot(x, y, 'bo')
plt.grid()
plt.show()