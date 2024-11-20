from gensim.models import Word2Vec

sentences = [["I", "love", "natural", "language", "processing"],
             ["Python", "is", "great", "for", "machine", "learning"],
             ["Deep", "learning", "is", "a", "subset", "of", "AI"]]

# Word2Vec 모델 학습
model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, sg=1)

# "learning"과 "machine"의 유사도 계산
similarity = model.wv.similarity("learning", "machine")
print("Similarity:", similarity)