import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics # 평가를 위한 모듈
import os
# print(os.getcwd())

df = pd.read_csv('bmi_500.csv', index_col='Label')
df.head()
df.info()
df.index.unique()

def easy_scatter(label, color):
  t = df.loc[label]
  plt.scatter(t['Weight'], t['Height'], color=color, label=label)

plt.figure( figsize=(5, 5) )

easy_scatter('Extreme Obesity', 'black')
easy_scatter('Weak', 'blue')
easy_scatter('Normal', 'green')
easy_scatter('Overweight', 'pink')
easy_scatter('Obesity', 'purple')
easy_scatter('Extremely Weak', 'red')

plt.legend()
plt.show()