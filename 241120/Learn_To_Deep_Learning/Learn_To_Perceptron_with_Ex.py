import numpy as np

x = np.array([0, 1])      # 입력
w = np.array([0.5 , 0.5]) # 가중치: 입력 신호가 결과에 주는 영향력(중요도)을 조절하는 매개변수
b = -0.7   # 편향: 뉴런이 얼마나 쉽게 활성화(결과로 1을 출력) 하느냐를 조정하는 매개변수

print(w*x)
print( np.sum(w*x) )
print( np.sum(w*x) + b )
def AND(x1, x2):
  x = np.array([x1, x2])    # 입력
  w = np.array([0.5 , 0.5]) # 가중치
  b = -0.7                  # 편향

  tmp = np.sum(w*x) + b

  if tmp <= 0:
    return 0
  else:
    return 1

print(AND(0, 0))
print(AND(0, 1))
print(AND(1, 0))
print(AND(1, 1))