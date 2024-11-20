from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 문서 리스트
documents = [
    "I love data science and machine learning.",
    "Machine learning is a part of data science.",
    "Deep learning advances machine learning."
]

# TF-IDF 계산
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# 코사인 유사도 계산
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print("코사인 유사도:\n", cosine_sim)