# Sentence-BERT로 임베딩한 문장을 K-Means 클러스터링
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

# 문장 데이터
sentences = ["I love NLP.", "Deep learning is great.", "I enjoy Python.", "Clustering is fun."]

# 문장 임베딩
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(sentences)

# K-Means 클

# 러스터링
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(embeddings)

print("클러스터 레이블:", labels)