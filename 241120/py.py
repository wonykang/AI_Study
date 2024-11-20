import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

# 학습시간 대비 합격 데이터
pass_time = [8, 9, 9, 9.5, 10, 12, 14, 14.5, 15, 16, 16, 16.5, 17, 17, 17, 17.5, 20, 20, 20]
fail_time = [1, 2, 2.1, 2.6, 2.7, 2.8, 2.9, 3, 3.2, 3.4, 3.5, 3.6, 3, 5, 5.2, 5.4]

# X
X = np.hstack( (pass_time, fail_time) )
X
X.shape, X.reshape(-1, 1).shape
# y
y1 = [1] * len(pass_time)
y0 = [0] * len(fail_time)
y = np.hstack( (y1, y0) )
y
# 모델학습
model = LogisticRegression()
model.fit(X.reshape(-1, 1), y)
model.coef_, model.intercept_


model.predict( [ [7.0], [6.9] ] )
model.predict_proba( [ [7.0], [6.9] ] )

# 모델시각화
def logreg(z):
  return 1 / (1 + np.exp(-z))

xx = np.linspace(1, 21, 100)
yy = logreg(model.coef_ * xx + model.intercept_)[0]
print(yy.shape)

plt.plot(xx, yy, c='r')
plt.scatter(X, y)
plt.grid()
plt.show()

# # 가중치(기울기) 변화에 따른 시각화
# w_list = [0.3, 0.5, 1.0]  # 가중치
# b_list = [0]              # 편향
# xx = np.linspace(-10, 10, 100)
# for w in w_list:
#   for b in b_list:
#     yy = logreg( w * xx + b )
#     plt.plot(xx, yy, label=f'{w}')

# plt.legend()

# 편향(절편) 변화에 따른 시각화
w_list = [0.8]         # 가중치
b_list = [-2, 0, 2]    # 편향
xx = np.linspace(-10, 10, 100)
for w in w_list:
  for b in b_list:
    yy = logreg( w * xx + b )
    plt.plot(xx, yy, label=f'{b}')

plt.legend()