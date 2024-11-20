import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

n = 100
x = 6 * np.random.rand(n, 1) - 3
y = 0.5 * x**2 + 1 * x + 2 + np.random.rand(n, 1)

plt.scatter(x, y, s=5)
plt.show()