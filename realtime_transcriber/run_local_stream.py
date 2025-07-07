from transcriber import StreamingTranscriber
from vad import split_audio_to_chunks
from config import vad, SAMPLE_RATE, CHUNK_DURATION
import torchaudio
import time

from ConsoleStreamer import ConsoleStreamer

def main():
    streamer = ConsoleStreamer()
    transcriber = StreamingTranscriber(vad)

    waveform, sr = torchaudio.load("shorts.wav")
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    audio = waveform[0].numpy()
    chunks = split_audio_to_chunks(audio, SAMPLE_RATE, CHUNK_DURATION)

    print("\n" * 2)  # 預留兩行空白做覆蓋用

    for chunk in chunks:
        result = transcriber.process_chunk(chunk)
        if result:
            streamer.display(result)
        time.sleep(CHUNK_DURATION)

if __name__ == "__main__":
    main()
