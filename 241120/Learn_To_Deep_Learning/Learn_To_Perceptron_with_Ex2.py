import numpy as np

def OR(x1, x2):
  x = np.array([x1, x2])    # 입력
  w = np.array([0.5 , 0.5]) # 가중치
  b = -0.2                # 편향

  tmp = np.sum(w*x) + b

  if tmp <= 0:
    return 0
  else:
    return 1

print(OR(0, 0))
print(OR(0, 1))
print(OR(1, 0))
print(OR(1, 1))