import torchaudio
import time
import sounddevice as sd
import sys
from config import SAMPLE_RATE, CHUNK_DURATION, vad
from vad import split_audio_to_chunks
from transcriber_openai import StreamingTranscriber

def main():
    wav_path = "./shorts.wav"
    waveform, sr = torchaudio.load(wav_path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    audio = waveform[0].numpy()
    chunks = split_audio_to_chunks(audio, SAMPLE_RATE, CHUNK_DURATION)

    transcriber = StreamingTranscriber(vad)

    print("🔊 開始播放與即時辨識...\n")

    interim_text = ""
    interim_active = False

    for chunk in chunks:
        # 播放這段音訊
        sd.play(chunk, samplerate=SAMPLE_RATE)
        
        # 並行辨識
        result = transcriber.process_chunk(chunk)

        if result:
            if result["type"] == "interim":
                interim_text += result["text"]
                sys.stdout.write("\r")
                sys.stdout.write(f"⏳ [暫定] {interim_text[:80]}")
                sys.stdout.flush()
                interim_active = True

            elif result["type"] == "final":
                if interim_active:
                    sys.stdout.write("\r" + " " * 100 + "\r")
                print(f"✅ [修正] ({result['speaker']}) {result['start']}~{result['end']}s: {result['text']}")
                interim_text = ""
                interim_active = False

        # 等待播放完畢（非強制等，但較穩定）
        sd.wait()

if __name__ == "__main__":
    main()
