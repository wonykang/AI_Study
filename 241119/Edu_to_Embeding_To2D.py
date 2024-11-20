from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 단어와 벡터
words = ["king", "queen", "man", "woman"]
vectors = [
    [0.5, 0.8, 0.3],
    [0.4, 0.7, 0.3],
    [0.5, 0.9, 0.2],
    [0.4, 0.6, 0.3]
]

# PCA로 차원 축소
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(vectors)

# 시각화
plt.figure(figsize=(8, 6))
for word, coord in zip(words, reduced_vectors):
    plt.scatter(coord[0], coord[1], label=word)
    plt.text(coord[0] + 0.02, coord[1] + 0.02, word, fontsize=12)

plt.title("Word Embedding Visualization")
plt.grid()
plt.legend()
plt.show()