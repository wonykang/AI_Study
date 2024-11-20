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
    bigram: count / unigram_counts[bigram[0]]
    for bigram, count in bigram_counts.items()
}

print("2-그램 확률:")
for bigram, prob in bigram_probabilities.items():
    print(f"P({bigram[1]} | {bigram[0]}) = {prob:.4f}")