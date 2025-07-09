import torchaudio
import time
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

from configs.config import SAMPLE_RATE, CHUNK_DURATION, vad
from utils.vad import split_audio_to_chunks
from transcriber.transcriber import StreamingTranscriber

# 正確指定 static 目錄
app = Flask(__name__, static_url_path="/static", static_folder="static")
socketio = SocketIO(app, async_mode="threading")  # 更穩定

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("start_transcription")
def handle_transcription():
    wav_path = "./static/test.mp3"
    waveform, sr = torchaudio.load(wav_path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    audio = waveform[0].numpy()
    chunks = split_audio_to_chunks(audio, SAMPLE_RATE, CHUNK_DURATION)

    transcriber = StreamingTranscriber(vad)

    for chunk in chunks:
        result = transcriber.process_chunk(chunk)

        if result:
            if result["type"] == "interim":
                emit("interim", result["text"])
            elif result["type"] == "final":
                emit("final", {
                    "speaker": result["speaker"],
                    "start": result["start"],
                    "end": result["end"],
                    "text": result["text"]
                })

        time.sleep(CHUNK_DURATION)

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8946, debug=True)
