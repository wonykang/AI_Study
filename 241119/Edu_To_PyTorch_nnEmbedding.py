import torch
import torch.nn as nn

# 데이터
vocab_size = 10
embedding_dim = 8
input_data = torch.tensor([[1, 2, 0], [3, 4, 0]])

# 임베딩 레이어 정의
embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

# 임베딩 결과
output = embedding(input_data)
print("임베딩 결과:", output)