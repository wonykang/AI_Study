import nltk
import konlpy
import torch
import re

text = "PyTorch makes it easier to work with deep learning models."

# # 공백 기준 토큰화
# tokens = text.split()
# print("Tokens:", tokens)

# 정규 표현식을 사용한 토큰화
tokens = re.findall(r'\b\w+\b', text)
print("Tokens:", tokens)
# 출력: ['PyTorch', 'makes', 'it', 'easier', 'to', 'work', 'with', 'deep', 'learning', 'models']