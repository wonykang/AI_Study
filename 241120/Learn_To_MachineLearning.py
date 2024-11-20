# pandas 라이브러리를 사용하여 코드 간략화
import pandas as pd
from sklearn import svm, metrics

# XOR 연산
xor_input = [
   [0, 0, 0],
   [0, 1, 1],
   [1, 0, 1],
   [1, 1, 0]
]

# 입력을 학습 전용 데이터와 테스트 전용 데이터로 분류하기 --- (※1)
xor_df = pd.DataFrame(xor_input)
xor_data = xor_df[ [0, 1] ]
xor_label = xor_df[2]

# 데이터 학습과 예측하기 --- (※2)
model = svm.SVC()
model.fit(xor_data, xor_label)
pre = model.predict(xor_data)

# 정답률 구하기 --- (※3)
ac_score = metrics.accuracy_score(xor_label, pre)
print(ac_score)