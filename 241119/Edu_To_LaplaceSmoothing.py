from collections import Counter

# 주어진 빈도수
bigram_counts = {
    ('I', 'love'): 2,
    ('love', 'NLP'): 1
}
unigram_counts = {
    'I': 3,
    'love': 2,
    'NLP': 1
}
vocab_size = 5  # 단어 집합 크기

# 라플라스 스무딩 확률 계산
def smoothed_bigram_probability(w1, w2):
    count_bigram = bigram_counts.get((w1, w2), 0)
    count_unigram = unigram_counts.get(w1, 0)
    return (count_bigram + 1) / (count_unigram + vocab_size)

print(f"P(love | I): {smoothed_bigram_probability('I', 'love'):.4f}")
print(f"P(NLP | love): {smoothed_bigram_probability('love', 'NLP'):.4f}")