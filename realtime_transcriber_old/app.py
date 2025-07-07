from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import torchaudio
from transcriber import process_audio_stream
from config import vad

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        wav_path = "./shorts.wav"
        waveform, sr = torchaudio.load(wav_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)

        audio = waveform[0].numpy()
        chunk_size = int(0.5 * 16000)
        chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]

        await process_audio_stream(chunks, vad, websocket)
        await websocket.close()

    except Exception as e:
        print("❌ WebSocket Error:", e)
        await websocket.close()
