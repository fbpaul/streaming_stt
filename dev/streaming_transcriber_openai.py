import os
import time
import tempfile
import numpy as np
import webrtcvad
import torchaudio
from pyannote.audio import Pipeline
from dotenv import load_dotenv
from scipy.io.wavfile import write as wav_write
from openai import OpenAI

# -------------------
# 參數設定
# -------------------
CHUNK_DURATION = 0.5  # 每段 0.5 秒
SAMPLE_RATE = 16000
VAD_MODE = 3          # 最嚴格
MIN_SPEECH_DURATION = 1.5  # 超過 1.5 秒才送出辨識
LANGUAGE = "zh"

# -------------------
# 環境變數與模型初始化
# -------------------
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
hf_token = os.getenv("HUGGINGFACE_TOKEN")
os.environ["HF_TOKEN"] = hf_token
vad = webrtcvad.Vad(VAD_MODE)
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)

# -------------------
# 工具函數
# -------------------
def load_audio_chunks(wav_path):
    waveform, sr = torchaudio.load(wav_path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    audio = waveform[0].numpy()
    chunk_size = int(CHUNK_DURATION * SAMPLE_RATE)
    return [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]

def float32_to_pcm16(audio_float):
    audio_float = np.clip(audio_float, -1, 1)
    return (audio_float * 32767).astype(np.int16).tobytes()

def is_speech(chunk, sample_rate=SAMPLE_RATE, threshold=0.5):
    """
    判斷 chunk 中語音比例是否超過 threshold。
    回傳 True 表示是語音段，False 表示不是。
    """
    if not isinstance(chunk, np.ndarray) or chunk.dtype != np.float32:
        raise ValueError("Input chunk must be float32 numpy array.")

    pcm = float32_to_pcm16(chunk)

    frame_duration = 30  # 每小段 30ms
    frame_size = int(sample_rate * frame_duration / 1000)  # 480 samples
    byte_size = frame_size * 2  # 每 sample 2 bytes

    num_frames = len(pcm) // byte_size
    if num_frames == 0:
        return False

    speech_count = 0

    for i in range(num_frames):
        start = i * byte_size
        end = start + byte_size
        frame = pcm[start:end]
        if len(frame) < byte_size:
            continue
        try:
            if vad.is_speech(frame, sample_rate):
                speech_count += 1
        except webrtcvad.Error:
            continue  # 避免 VAD 異常造成中斷

    speech_ratio = speech_count / num_frames
    # print(f"🔍 語音比例: {speech_ratio:.2f}")
    return speech_ratio >= threshold


def transcribe_streaming(chunks):
    buffer = []
    sentence_id = 0

    for i, chunk in enumerate(chunks):
        print(f"⏳ 模擬串流傳入第 {i+1} 段...")

        if is_speech(chunk):
            buffer.append(chunk)
        else:
            if len(buffer) * CHUNK_DURATION >= MIN_SPEECH_DURATION:
                sentence_id += 1
                audio_segment = np.concatenate(buffer)
                buffer = []

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmpfile:
                    wav_write(tmpfile.name, SAMPLE_RATE, (audio_segment * 32768).astype(np.int16))

                    with open(tmpfile.name, "rb") as f:
                        response = openai_client.audio.transcriptions.create(
                            model="whisper-1",
                            file=f,
                            language=LANGUAGE
                        )
                        print(f"\n🗨 第 {sentence_id} 句轉譯結果：{response.text.strip()}")
            else:
                buffer = []

        time.sleep(CHUNK_DURATION)

def speaker_diarization(wav_path):
    print("\n🧑‍🤝‍🧑 語者分離分析中...")
    result = diarization_pipeline(wav_path)
    for turn, _, speaker in result.itertracks(yield_label=True):
        print(f"{turn.start:.1f}s - {turn.end:.1f}s: {speaker}")

# -------------------
# 主程式
# -------------------
def main():
    wav_path = "./shorts.wav"
    print("🔊 開始模擬串流語音辨識...\n")
    chunks = load_audio_chunks(wav_path)
    transcribe_streaming(chunks)
    speaker_diarization(wav_path)

if __name__ == "__main__":
    main()
