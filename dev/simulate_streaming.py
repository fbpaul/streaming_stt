import os
import io
import time
import webrtcvad
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel

# 初始化語者分離 pipeline（需 HuggingFace Token）
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN") or "hf_XXXX"  # <<-- 替換為你的 token
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HUGGINGFACE_TOKEN)

# 初始化 Whisper 模型（支援中文）
whisper_model = WhisperModel("medium", device="cuda", compute_type="float16")

# 初始化 VAD（WebRTC）
vad = webrtcvad.Vad()
vad.set_mode(2)  # 0: aggressive, 3: very aggressive

# 模擬輸入音檔切成每 200ms 丟進 buffer（可改為你的串流來源）
AUDIO_PATH = "sample_zh.wav"  # 你的中文語音檔
original = AudioSegment.from_wav(AUDIO_PATH)
frame_ms = 200
frame_bytes = int(original.frame_rate * (frame_ms / 1000.0) * 2)

buffer = []
silent_counter = 0
trigger_silence_threshold = 5

print("🔁 開始模擬串流輸入...")

for i in range(0, len(original), frame_ms):
    chunk = original[i:i+frame_ms]
    raw_audio = chunk.raw_data
    is_speech = vad.is_speech(raw_audio, sample_rate=original.frame_rate)

    if is_speech:
        buffer.append(chunk)
        silent_counter = 0
    else:
        silent_counter += 1
        if buffer:
            buffer.append(chunk)

    # 如果連續靜音段超過門檻，觸發語音推論
    if silent_counter >= trigger_silence_threshold and buffer:
        combined = sum(buffer)
        wav_io = io.BytesIO()
        combined.export(wav_io, format="wav")
        wav_io.seek(0)
        audio_data, _ = sf.read(wav_io)

        # 儲存暫存檔供 pyannote 使用
        with open("temp_stream.wav", "wb") as f:
            combined.export(f, format="wav")

        print("🧠 開始語者分離與辨識...")

        diarization = diarization_pipeline("temp_stream.wav")

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            seg = combined[turn.start * 1000: turn.end * 1000]
            seg_io = io.BytesIO()
            seg.export(seg_io, format="wav")
            seg_io.seek(0)
            audio_clip, _ = sf.read(seg_io)

            segments, _ = whisper_model.transcribe(audio_clip, language="zh", vad_filter=False)
            full_text = " ".join([seg.text for seg in segments])
            print(f"🗣️ {speaker}: {full_text}")

        # 清空 buffer
        buffer = []
        silent_counter = 0

    time.sleep(0.05)  # 模擬串流間隔

print("✅ 模擬結束")
