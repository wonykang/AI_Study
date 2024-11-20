import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

texts = ["I enjoy programming", "Programming is fun"]

# 원-핫 인코딩
tokens = [set(text.lower().split()) for text in texts]
vocab = sorted(set(word for text in tokens for word in text))
one_hot_vectors = [[1 if word in text else 0 for word in vocab] for text in tokens]

# 코사인 유사도 계산
cosine_sim = cosine_similarity(one_hot_vectors)
print("코사인 유사도:\n", cosine_sim)