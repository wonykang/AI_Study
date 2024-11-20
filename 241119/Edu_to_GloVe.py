import numpy as np

# GloVe 벡터 로드
def load_glove(filepath):
    embeddings = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = np.array(parts[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

glove_path = "glove.6B.100d.txt"
embeddings = load_glove(glove_path)

# 벡터 연산
result = embeddings["king"] - embeddings["man"] + embeddings["woman"]

# 가장 유사한 단어 찾기
similarity = {word: np.dot(result, vec) for word, vec in embeddings.items()}
predicted_word = max(similarity, key=similarity.get)
print("Result:", predicted_word)