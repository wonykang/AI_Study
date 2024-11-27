from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoTokenizer,AutoModel
from datasets import load_dataset
from transformers import pipeline
import torch

model_name = "bert-base-uncased"

# BERT 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 입력 문장
text = "Hugging Face makes NLP easy!"

# 텍스트 토큰화
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

print(inputs)
# 출력 예: {'input_ids': tensor([[...]]), 'attention_mask': tensor([[...]])}