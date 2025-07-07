import os
import numpy as np
import webrtcvad
import torchaudio
import torch
from pyannote.audio import Pipeline
# from faster_whisper import WhisperModel
import time

# -------------------
# 參數設定
# -------------------
CHUNK_DURATION = 0.5  # 每次丟 0.5 秒模擬串流
SAMPLE_RATE = 16000
VAD_MODE = 3  # 最嚴格語音偵測
LANGUAGE = "zh"

# -------------------
# 初始化模型
# -------------------
asr_model = WhisperModel("medium", device="cuda", compute_type="float16")
vad = webrtcvad.Vad(VAD_MODE)
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                                use_auth_token=os.getenv("HUGGINGFACE_TOKEN"))

# -------------------
# 載入與切割音訊
# -------------------
def load_audio_and_chunk(wav_path):
    waveform, sr = torchaudio.load(wav_path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    audio = waveform[0].numpy()

    chunk_size = int(CHUNK_DURATION * SAMPLE_RATE)
    return [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]

# -------------------
# 判斷是否為語音
# -------------------
def is_speech(chunk):
    pcm = (chunk * 32768).astype(np.int16).tobytes()
    return vad.is_speech(pcm, SAMPLE_RATE)

# -------------------
# 模擬即時語音轉換流程
# -------------------
def transcribe_streaming(chunks):
    buffer = []
    sentence_id = 0

    for i, chunk in enumerate(chunks):
        if is_speech(chunk):
            buffer.append(chunk)
        else:
            # 如果累積的語音長度超過1.5秒，則進行轉換
            if len(buffer) * CHUNK_DURATION >= 1.5:
                audio_segment = np.concatenate(buffer)
                segments, _ = asr_model.transcribe(audio_segment, language=LANGUAGE)

                print(f"\n🗨 第 {sentence_id + 1} 句轉譯結果：")
                for seg in segments:
                    print(f"{seg.text.strip()}")

                buffer = []
                sentence_id += 1
            else:
                buffer = []

# -------------------
# 語者分離
# -------------------
def speaker_diarization(wav_path):
    result = diarization_pipeline(wav_path)
    print("\n🧑‍🤝‍🧑 語者分離結果:")
    for turn, _, speaker in result.itertracks(yield_label=True):
        print(f"{turn.start:.1f}s - {turn.end:.1f}s: {speaker}")

# -------------------
# 主程式
# -------------------
def main():
    wav_path = "example.wav"  # 改成你的檔名
    chunks = load_audio_and_chunk(wav_path)

    print("🔊 開始模擬串流語音辨識...\n")
    transcribe_streaming(chunks)
    speaker_diarization(wav_path)

if __name__ == "__main__":
    main()
