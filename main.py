import torchaudio
import time
import sys

from configs.config import SAMPLE_RATE, CHUNK_DURATION, vad, FILE
from utils.vad import split_audio_to_chunks
# from transcriber.transcriber_openai import StreamingTranscriber
from transcriber.transcriber import StreamingTranscriber

def main():
    wav_path = FILE
    waveform, sr = torchaudio.load(wav_path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    audio = waveform[0].numpy()
    chunks = split_audio_to_chunks(audio, SAMPLE_RATE, CHUNK_DURATION)

    transcriber = StreamingTranscriber(vad)

    print("🔊 開始模擬串流辨識...\n")
    interim_text = ""
    interim_active = False

    for i, chunk in enumerate(chunks):
        # print(f"處理第 {i+1} 個片段...")
        result = transcriber.process_chunk(chunk)
        if result:
            if result["type"] == "interim":
                interim_text += result["text"]
                sys.stdout.write("\r")  # 回到行首
                sys.stdout.write(f"⏳ [暫定] {interim_text[:80]}")  # 顯示前 80 字
                sys.stdout.flush()
                interim_active = True

            elif result["type"] == "final":
                if interim_active:
                    sys.stdout.write("\r" + " " * 100 + "\r")  # 清除暫定行
                print(f"✅ [修正] ({result['speaker']}) {result['start']}~{result['end']}s: {result['text']}")
                interim_text = ""
                interim_active = False

        time.sleep(CHUNK_DURATION)

if __name__ == "__main__":
    main()
