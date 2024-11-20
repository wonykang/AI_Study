import numpy as np

# 주어진 2-그램 확률
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

# 퍼플렉서티 계산
perplexity = np.power(1 / probability, 1 / (len(tokens) - 1))
print(f"퍼플렉서티: {perplexity:.4f}")