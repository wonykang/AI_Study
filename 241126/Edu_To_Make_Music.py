import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import mido
from mido import MidiFile, MidiTrack, Message


class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, hidden_size):
        super(MusicTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Transformer 모델의 batch_first=True로 설정
        self.transformer = nn.Transformer(
            d_model=embed_size, 
            nhead=num_heads, 
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_size,
            batch_first=True  # batch_first=True로 설정
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, src, tgt):
        # src, tgt: (batch_size, sequence_length)
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)
        
        transformer_out = self.transformer(src_emb, tgt_emb)
        output = self.fc_out(transformer_out)
        return output

# 모델 초기화
vocab_size = 128  # MIDI 음표 수 (예시)
embed_size = 256
num_heads = 8
num_layers = 6
hidden_size = 512

model = MusicTransformer(vocab_size, embed_size, num_heads, num_layers, hidden_size)
model.eval()  # 모델을 평가 모드로 설정

# 디렉토리 생성 함수
def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# MIDI 파일 생성 함수 (K-pop 스타일 빠른 변주곡)
def create_midi_from_sequence(sequence, output_file='generated_music.mid'):
    # 파일을 저장할 디렉토리
    output_directory = os.path.dirname(output_file)

    # 디렉토리가 존재하지 않으면 생성
    create_directory_if_not_exists(output_directory)
    
    # 새로운 MIDI 파일을 만듭니다.
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)
    
    # 템포 설정 (빠른 템포, 예: 140 BPM)
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(140)))  # 140 BPM
    
    # 기본 코드 진행 (C, G, Am, F)
    chords = [
        [60, 64, 67],  # C major (C, E, G)
        [55, 59, 62],  # G major (G, B, D)
        [57, 60, 64],  # A minor (A, C, E)
        [53, 57, 60]   # F major (F, A, C)
    ]
    
    # 트랙에 음표 추가 (음표 길이는 짧게, 빠른 변주곡 느낌)
    time_per_note = 120  # 빠르게 전개되는 음표 간격
    
    # 첫 번째 코드를 추가 (C major)
    for chord in chords:
        for note in chord:
            track.append(Message('note_on', note=note, velocity=64, time=0))
        for note in chord:
            track.append(Message('note_off', note=note, velocity=64, time=time_per_note))
    
    # 빠르게 변주할 음표들을 생성
    for _ in range(8):  # 8번 반복해서 변주
        # 랜덤으로 다음 코드를 선택하여 변주
        chord = random.choice(chords)
        
        # 음표를 변주 (같은 코드지만 랜덤하게 변주)
        for note in chord:
            track.append(Message('note_on', note=note, velocity=64, time=0))
        for note in chord:
            track.append(Message('note_off', note=note, velocity=64, time=time_per_note))
        
        # 일부 음표는 반음 이동하여 변형 (예: 반음 올리기)
        shifted_chord = [note + random.choice([0, 1]) for note in chord]  # 반음씩 올려서 변형
        
        for note in shifted_chord:
            track.append(Message('note_on', note=note, velocity=64, time=0))
        for note in shifted_chord:
            track.append(Message('note_off', note=note, velocity=64, time=time_per_note))

    # 빠른 템포에 맞춰 리듬 변화
    track.append(Message('note_on', note=60, velocity=64, time=0))  # C4
    track.append(Message('note_off', note=60, velocity=64, time=60))  # 1비트 길이
    track.append(Message('note_on', note=62, velocity=64, time=0))  # D4
    track.append(Message('note_off', note=62, velocity=64, time=60))  # 1비트 길이
    track.append(Message('note_on', note=64, velocity=64, time=0))  # E4
    track.append(Message('note_off', note=64, velocity=64, time=60))  # 1비트 길이

    # MIDI 파일 저장
    midi.save(output_file)
    print(f"MIDI 파일 '{output_file}'가 생성되었습니다.")

# 예시: 생성된 MIDI 값 시퀀스 (K-pop 스타일 빠른 변주곡)
generated_sequence = [60, 64, 67, 55, 59, 62, 57, 60, 64, 53, 57, 60]  # C, G, Am, F 코드

# 저장할 경로 지정 (예시: Desktop의 'music_output' 폴더에 저장)
output_path = r'C:\Users\UserC\Desktop\AILLM\music_output\generated_music.mid'

# 음악을 MIDI 파일로 저장
create_midi_from_sequence(generated_sequence, output_file=output_path)