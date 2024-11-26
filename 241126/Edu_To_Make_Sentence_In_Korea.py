import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

# KoGPT-2 모델과 토크나이저 로드
model_name = "skt/kogpt2-base-v2"  # KoGPT-2 모델을 사용
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)  # KoGPT-2에 맞는 토크나이저 사용

# pad_token이 None인 경우 eos_token을 pad_token으로 설정
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # '[PAD]' 토큰 추가
    model.config.pad_token_id = tokenizer.pad_token_id  # 모델에 패딩 토큰 ID 설정


# 패딩 토큰 확인 (디버깅 로그)
print("패딩 토큰:", tokenizer.pad_token)  # 설정된 pad_token 확인
print("패딩 토큰 ID:", tokenizer.pad_token_id)  # 설정된 pad_token_id 확인

# 모델을 평가 모드로 설정
model.eval()

# 사용자로부터 한국어 키워드 입력 받기
keywords = input("콤마로 구분된 한국어 키워드를 입력하세요: ")

# 프롬프트 개선
input_text = f"다음 키워드들을 사용하여 관련성이 있는 문장을 만들어 주세요: {keywords}. 이 문장은 특정 주제에 대한 설명이나 이야기를 포함해야 합니다."

# 입력 텍스트를 토큰화하여 모델 입력 형식으로 변환
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=50)

# 텍스트 생성
with torch.no_grad():
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=150,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        do_sample=True
    )

# 생성된 텍스트 디코딩
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 후처리: 불필요한 기호 제거 및 문장 다듬기
generated_text = generated_text.replace('[]', '').strip()
print("생성된 문장:", generated_text)