import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Device 설정 (GPU 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 샘플 영어-프랑스 병렬 데이터
data = [
    ("i am a student", "je suis un étudiant"),
    ("he is a teacher", "il est un professeur"),
    ("she is a doctor", "elle est un médecin"),
    ("we are friends", "nous sommes amis"),
    ("they are happy", "ils sont heureux"),
]

# 사전 생성
def build_vocab(sentences):
    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
    for sentence in sentences:
        for word in sentence.split():
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab

src_sentences = [pair[0] for pair in data]
tgt_sentences = [pair[1] for pair in data]

src_vocab = build_vocab(src_sentences)
tgt_vocab = build_vocab(tgt_sentences)

# 토큰화 함수
def tokenize(sentence, vocab):
    tokens = ["<sos>"] + sentence.split() + ["<eos>"]
    return [vocab.get(token, vocab["<unk>"]) for token in tokens]

# 데이터셋 클래스
class TranslationDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_sentence, tgt_sentence = self.data[idx]
        src_tokens = tokenize(src_sentence, self.src_vocab)
        tgt_tokens = tokenize(tgt_sentence, self.tgt_vocab)
        return torch.tensor(src_tokens), torch.tensor(tgt_tokens)

dataset = TranslationDataset(data, src_vocab, tgt_vocab)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: collate_fn(x, src_vocab, tgt_vocab))

# 패딩 처리 함수
def collate_fn(batch, src_vocab, tgt_vocab):
    src_seqs, tgt_seqs = zip(*batch)
    src_seqs = nn.utils.rnn.pad_sequence(src_seqs, padding_value=src_vocab["<pad>"], batch_first=True)
    tgt_seqs = nn.utils.rnn.pad_sequence(tgt_seqs, padding_value=tgt_vocab["<pad>"], batch_first=True)
    return src_seqs, tgt_seqs

# 트랜스포머 모델
class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_encoder_layers, num_decoder_layers, d_ff, max_seq_len, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=d_ff,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        # 임베딩 + 포지셔널 인코딩
        src = self.src_embedding(src) + self.positional_encoding[:, :src.size(1), :]
        tgt = self.tgt_embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]
        # 트랜스포머 모델 실행
        output = self.transformer(src.transpose(0, 1), tgt.transpose(0, 1))
        output = self.fc_out(output.transpose(0, 1))
        return output

# 하이퍼파라미터 설정
src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)
d_model = 64
num_heads = 4
num_encoder_layers = 2
num_decoder_layers = 2
d_ff = 256
max_seq_len = 20
dropout = 0.1

# 모델 초기화
model = TransformerModel(
    src_vocab_size, tgt_vocab_size, d_model, num_heads, num_encoder_layers, num_decoder_layers, d_ff, max_seq_len, dropout
).to(device)

# 손실 함수와 옵티마이저
criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab["<pad>"])
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
def train(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            output = model(src, tgt_input)
            output = output.reshape(-1, output.size(-1))
            tgt_output = tgt_output.reshape(-1)
            loss = criterion(output, tgt_output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

train(model, dataloader, criterion, optimizer)

# 번역 함수
def translate(model, sentence, src_vocab, tgt_vocab, max_len=20):
    model.eval()
    src_tokens = tokenize(sentence, src_vocab)
    src_tensor = torch.tensor(src_tokens).unsqueeze(0).to(device)
    tgt_tokens = [tgt_vocab["<sos>"]]
    for _ in range(max_len):
        tgt_tensor = torch.tensor(tgt_tokens).unsqueeze(0).to(device)
        output = model(src_tensor, tgt_tensor)
        next_token = output.argmax(2)[:, -1].item()
        tgt_tokens.append(next_token)
        if next_token == tgt_vocab["<eos>"]:
            break
    tgt_sentence = " ".join([k for k, v in tgt_vocab.items() if v in tgt_tokens])
    return tgt_sentence

# 번역 예제
sentence = "i am a student"
translation = translate(model, sentence, src_vocab, tgt_vocab)
print(f"Input: {sentence}")
print(f"Translation: {translation}")