import torch
import torch.nn as nn
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


# MIDI 파일 생성 함수 (협주곡 스타일)
def create_midi_from_sequence(sequence, output_file='generated_music.mid'):
    # 파일을 저장할 디렉토리
    output_directory = os.path.dirname(output_file)

    # 디렉토리가 존재하지 않으면 생성
    create_directory_if_not_exists(output_directory)
    
    # 새로운 MIDI 파일을 만듭니다.
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)
    
    # 템포 설정 (아이유 스타일로 감정을 고조시키는 템포)
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(100)))  # 예시: 100 BPM

    # 코드 진행 (서정적인 코드)
    chords = [
        [60, 64, 67],  # C major (C, E, G)
        [55, 59, 62],  # G major (G, B, D)
        [57, 60, 64],  # A minor (A, C, E)
        [53, 57, 60],  # F major (F, A, C)
        [62, 66, 69],  # D minor (D, F, A)
    ]
    
    # 주악기: 피아노 (채널 0) - 코드 진행
    track.append(Message('program_change', program=0))  # 피아노 설정 (채널 0)
    time_per_note = 480  # 피아노 음표 길이
    for chord in chords:
        for note in chord:
            track.append(Message('note_on', note=note, velocity=64, time=0))
        for note in chord:
            track.append(Message('note_off', note=note, velocity=64, time=time_per_note))

    # 부악기: 기타 (채널 1) - 코드 진행
    track.append(Message('program_change', program=24))  # 기타 설정 (채널 1)
    for chord in chords:
        for note in chord:
            track.append(Message('note_on', note=note, velocity=64, time=0))
        for note in chord:
            track.append(Message('note_off', note=note, velocity=64, time=time_per_note))

    # 협주형 대화 추가: 피아노 (채널 0) + 기타 (채널 1) 간의 교차 대화
    for i in range(3):  # 3번 반복해서 교차 연주
        # 주악기: 피아노가 연주
        track.append(Message('program_change', program=0))  # 피아노
        track.append(Message('note_on', note=60, velocity=100, time=0))  # C4
        track.append(Message('note_off', note=60, velocity=100, time=480))  # 1 비트

        # 부악기: 기타가 응답
        track.append(Message('program_change', program=24))  # 기타
        track.append(Message('note_on', note=62, velocity=100, time=0))  # D4
        track.append(Message('note_off', note=62, velocity=100, time=480))  # 1 비트

    # 피아노와 기타가 함께 연주하는 클라이맥스
    track.append(Message('program_change', program=0))  # 피아노
    track.append(Message('note_on', note=60, velocity=100, time=0))  # C4
    track.append(Message('note_off', note=60, velocity=100, time=240))

    track.append(Message('program_change', program=24))  # 기타
    track.append(Message('note_on', note=60, velocity=100, time=0))  # C4
    track.append(Message('note_off', note=60, velocity=100, time=240))

    # 스트링(채널 2) - 협주 형식으로 배치
    track.append(Message('program_change', program=50))  # 스트링 설정 (채널 2)
    track.append(Message('note_on', note=64, velocity=100, time=0))  # E4
    track.append(Message('note_off', note=64, velocity=100, time=240))

    # 협주곡 형태의 최종 클라이맥스 (피아노, 기타, 스트링 모두 함께 연주)
    track.append(Message('program_change', program=0))  # 피아노
    track.append(Message('note_on', note=60, velocity=100, time=0))  # C4
    track.append(Message('note_off', note=60, velocity=100, time=240))

    track.append(Message('program_change', program=24))  # 기타
    track.append(Message('note_on', note=62, velocity=100, time=0))  # D4
    track.append(Message('note_off', note=62, velocity=100, time=240))

    track.append(Message('program_change', program=50))  # 스트링
    track.append(Message('note_on', note=64, velocity=100, time=0))  # E4
    track.append(Message('note_off', note=64, velocity=100, time=240))

    # MIDI 파일 저장
    midi.save(output_file)
    print(f"MIDI 파일 '{output_file}'가 생성되었습니다.")


# 예시: 협주곡 느낌을 주기 위한 랜덤 시퀀스 생성 함수
def generate_random_sequence(length=32):
    # 가능한 MIDI 음표 범위 (60~72)
    return [random.randint(60, 72) for _ in range(length)]  # 예: 60~72 (C4 ~ C5)

# 예시로 생성된 MIDI 값 시퀀스
generated_sequence = generate_random_sequence(length=32)

# 저장할 경로 지정 (예시: Desktop의 'music_output' 폴더에 저장)
output_path = r'C:\Users\UserC\Desktop\AILLM\music_output\generated_music_concerto.mid'

# 음악을 MIDI 파일로 저장
create_midi_from_sequence(generated_sequence, output_file=output_path)
