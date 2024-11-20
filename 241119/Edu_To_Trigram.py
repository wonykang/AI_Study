import numpy as np


# 주어진 빈도수
trigram_counts = {
    ('Deep', 'learning', 'advances'): 1,
    ('learning', 'advances', 'AI'): 1,
    ('advances', 'AI', 'research'): 1
}
bigram_counts = {
    ('Deep', 'learning'): 2,
    ('learning', 'advances'): 2,
    ('advances', 'AI'): 2
}

# 3-그램 확률 계산
def trigram_probability(w1, w2, w3):
    count_trigram = trigram_counts.get((w1, w2, w3), 0)
    count_bigram = bigram_counts.get((w1, w2), 0)
    return count_trigram / count_bigram if count_bigram > 0 else 0

print(f"P(advances | Deep, learning): {trigram_probability('Deep', 'learning', 'advances'):.4f}")