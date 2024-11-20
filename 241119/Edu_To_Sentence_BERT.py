from sentence_transformers import SentenceTransformer

# 모델 로드
model = SentenceTransformer("all-MiniLM-L6-v2")

# 문장 임베딩 계산
sentences = ["I love NLP.", "Word embeddings are powerful."]
embeddings = model.encode(sentences)
print("임베딩 결과:", embeddings)