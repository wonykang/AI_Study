import random

# 랜덤 삭제 함수
def random_deletion(sentence, p=0.3):
    words = sentence.split()
    if len(words) == 1:
        return sentence
    new_words = [word for word in words if random.random() > p]
    return ' '.join(new_words) if new_words else random.choice(words)

# 데이터 증강
sentence = "I love natural language processing"
augmented_sentence = random_deletion(sentence, p=0.3)

# 임베딩 계산 (Sentence-BERT 사용)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode(augmented_sentence)
print("Augmented Sentence:", augmented_sentence)
print("Embedding:", embedding)