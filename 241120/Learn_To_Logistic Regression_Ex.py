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

# y
y1 = [1] * len(pass_time)
y0 = [0] * len(fail_time)
y = np.hstack( (y1, y0) )
y
# 시각화
plt.scatter(X, y)
plt.grid()
plt.show()
