import torch
import torch.nn as nn
# 1. 회귀(Regression) 손실 함수
#  HingeEmbeddingLoss
loss = nn.HingeEmbeddingLoss(margin=1.0)


# 2. 분류(Classification) 손실 함수
#  KLDivLoss (Kullback-Leibler Divergence Loss)
loss = nn.KLDivLoss(reduction="batchmean")
# PoissonNLLLoss (Poisson Negative Log Likelihood Loss)
loss = nn.PoissonNLLLoss()


# 3. 순서 학습(Ranking) 손실 함수
#  TripletMarginLoss
loss = nn.TripletMarginLoss(margin=1.0, p=2)
# MultiLabelMarginLoss
loss = nn.MultiLabelMarginLoss()
# 4. 다중 클래스 분류 및 다중 레이블 손실 함수
#  MultiLabelSoftMarginLoss
loss = nn.MultiLabelSoftMarginLoss()
# MultiMarginLoss
loss = nn.MultiMarginLoss()
# 5. 시간순서 데이터(Time-Series) 손실 함수
# CTCLoss (Connectionist Temporal Classification Loss)
loss = nn.CTCLoss(blank=0)

# 6. 사용자 정의 손실 함수
