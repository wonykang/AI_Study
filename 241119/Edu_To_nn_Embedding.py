import torch
import torch.nn as nn

# 어휘 크기와 임베딩 차원 정의
vocab_size = 6
embedding_dim = 4

# Embedding 레이어 정의
embedding = nn.Embedding(vocab_size, embedding_dim)

# 단어 인덱스 입력
input_indices = torch.tensor([1, 2, 3, 4])

# 임베딩 생성
embedded = embedding(input_indices)
print("임베딩 결과:", embedded)