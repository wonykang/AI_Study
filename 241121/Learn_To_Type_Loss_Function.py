import torch.nn as nn
# 1. 회귀(Regression) 손실 함수
#  MSELoss (Mean Squared Error Loss)
loss = nn.MSELoss()
#  L1Loss (Mean Absolute Error Loss)
loss = nn.L1Loss()
# SmoothL1Loss (Huber Loss)
loss = nn.SmoothL1Loss()
# 2. 분류(Classification) 손실 함수
# CrossEntropyLoss
loss = nn.CrossEntropyLoss()
# NLLLoss (Negative Log Likelihood Loss)
loss = nn.NLLLoss()
# BCELoss (Binary Cross Entropy Loss)
loss = nn.BCELoss()
# BCEWithLogitsLoss
loss = nn.BCEWithLogitsLoss()
# 3. 순서 학습(Ranking) 손실 함수
# MarginRankingLoss
loss = nn.MarginRankingLoss()
# CosineEmbeddingLoss
loss = nn.CosineEmbeddingLoss()

# 6. 손실 함수 선택 기준
# 회귀 문제: MSELoss, L1Loss, SmoothL1Loss.
# 이진 분류: BCELoss, BCEWithLogitsLoss.
# 다중 클래스 분류: CrossEntropyLoss, NLLLoss.
# 순위 학습: MarginRankingLoss, CosineEmbeddingLoss.