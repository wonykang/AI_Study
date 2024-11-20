from gensim.models import KeyedVectors

# FastText 모델 로드
model_path = "cc.en.300.bin"  # 사전 학습된 FastText 모델 경로
fasttext = KeyedVectors.load_word2vec_format(model_path, binary=True)

# "king"과 가장 유사한 단어 찾기
most_similar = fasttext.most_similar("king", topn=1)
print("Most similar to 'king':", most_similar)