import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import torchaudio
import time

from vad import split_audio_to_chunks
from transcriber import StreamingTranscriber
from config import vad, SAMPLE_RATE, CHUNK_DURATION

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ Client connected.")

    # 載入音訊並切片
    wav_path = "./shorts.wav"
    waveform, sr = torchaudio.load(wav_path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    audio = waveform[0].numpy()
    chunks = split_audio_to_chunks(audio, SAMPLE_RATE, CHUNK_DURATION)

    transcriber = StreamingTranscriber(vad)

    # 模擬串流
    for chunk in chunks:
        result = transcriber.process_chunk(chunk)
        if result:
            await websocket.send_json(result)
        time.sleep(CHUNK_DURATION)

    await websocket.close()
    print("❌ Client disconnected.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
