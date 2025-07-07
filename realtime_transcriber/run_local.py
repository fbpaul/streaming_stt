import torchaudio
import time
from config import SAMPLE_RATE, CHUNK_DURATION, vad
from vad import split_audio_to_chunks
from transcriber import StreamingTranscriber

def main():
    # 載入音訊檔
    wav_path = "./shorts.wav"
    waveform, sr = torchaudio.load(wav_path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    audio = waveform[0].numpy()
    chunks = split_audio_to_chunks(audio, SAMPLE_RATE, CHUNK_DURATION)

    # 初始化 transcriber
    transcriber = StreamingTranscriber(vad)

    print("🔊 開始模擬串流辨識...\n")

    # 模擬串流逐段送入
    for i, chunk in enumerate(chunks):
        result = transcriber.process_chunk(chunk)
        if result:
            if result["type"] == "interim":
                print(f"⏳ [暫定] {result['text']}")
            elif result["type"] == "final":
                print(f"✅ [修正] ({result['speaker']}) {result['start']}~{result['end']}s: {result['text']}")
        time.sleep(CHUNK_DURATION)

if __name__ == "__main__":
    main()
