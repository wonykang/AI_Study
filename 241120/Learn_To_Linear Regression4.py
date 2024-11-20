import matplotlib.pyplot as plt
import numpy as np
import mglearn

X, y = mglearn.datasets.make_wave(100)
plt.scatter(X, y)
mglearn.plots.plot_linear_regression_wave()
plt.show()