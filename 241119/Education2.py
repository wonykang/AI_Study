from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

documents = [
    "Data science and machine learning are fun.",
    "Machine learning is a part of artificial intelligence.",
    "Deep learning advances AI and data science."
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# KMeans 군집화
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(tfidf_matrix)
print("문서 군집:", clusters)