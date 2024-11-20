from collections import Counter

# 문장
text = "I love natural language processing."

# 토큰화 및 2-그램 생성
tokens = text.split()
bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]

# 2-그램 및 단어 빈도수 계산
bigram_counts = Counter(bigrams)
unigram_counts = Counter(tokens)

# 2-그램 확률 계산
bigram_probabilities = {
    ('I', 'love'): 0.8,
    ('love', 'programming'): 0.6
}

# 문장
sentence = "I love programming"
tokens = sentence.split()

# 확률 계산
probability = 1
for i in range(len(tokens) - 1):
    bigram = (tokens[i], tokens[i+1])
    probability *= bigram_probabilities.get(bigram, 0)

print(f"문장의 확률: {probability:.4f}")