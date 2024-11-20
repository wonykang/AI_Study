import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics # 평가를 위한 모듈
# import os
# print(os.getcwd())

df = pd.read_csv('bmi_500.csv', index_col='Label')
df.head()

data = pd.read_csv('bmi_500.csv')

X = data.loc[:, 'Height':'Weight']
y = data.loc[:, 'Label']

print(X.shape)
print(y.shape)

X_train = X.iloc[:350, :]
X_test  = X.iloc[350:, :]
y_train = y.iloc[:350]
y_test  = y.iloc[350:]

bmi_model = KNeighborsClassifier(n_neighbors=10)
bmi_model.fit(X_train, y_train)
pre = bmi_model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, pre)
print('Accuracy:', accuracy)  # 정확도 출력