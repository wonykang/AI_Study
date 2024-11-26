import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 모델과 토크나이저 로드
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# 패딩 토큰 ID 설정 (GPT-2는 기본적으로 패딩 토큰이 없지만, 여기서는 None을 사용)
tokenizer.pad_token = tokenizer.eos_token  # 패딩 토큰을 eos_token으로 설정
model.config.pad_token_id = tokenizer.pad_token_id

# 모델을 평가 모드로 설정
model.eval()

# 텍스트 입력
input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=50)

# attention_mask 설정
attention_mask = inputs['attention_mask']

# 텍스트 생성
with torch.no_grad():
    outputs = model.generate(
        inputs['input_ids'],         # 입력 텍스트
        attention_mask=attention_mask,  # Attention Mask 추가
        max_length=50,               # 생성할 텍스트의 최대 길이
        num_return_sequences=1,      # 생성할 문장의 개수
        no_repeat_ngram_size=2,      # 반복 방지
        temperature=0.7,             # 온도 설정
        top_k=50,                    # top-k 샘플링
        top_p=0.95,                  # nucleus 샘플링
        do_sample=True               # 샘플링 방식
    )

# 생성된 텍스트 디코딩
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
