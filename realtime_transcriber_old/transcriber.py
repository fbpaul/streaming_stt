import tempfile
import numpy as np
from scipy.io.wavfile import write as wav_write
from config import openai_client, diarization_pipeline, SAMPLE_RATE, LANGUAGE, MIN_SPEECH_DURATION
from utils import is_speech

async def process_audio_stream(chunks, vad, websocket):
    buffer = []
    current_sentence = []

    for chunk in chunks:
        if is_speech(chunk, vad, SAMPLE_RATE):
            # 暫定文字：每段都送
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmpfile:
                wav_write(tmpfile.name, SAMPLE_RATE, (chunk * 32768).astype(np.int16))
                with open(tmpfile.name, "rb") as f:
                    response = openai_client.audio.transcriptions.create(model="whisper-1", file=f, language=LANGUAGE)
                    await websocket.send_json({"type": "interim", "text": response.text.strip()})

            current_sentence.append(chunk)

        else:
            if len(current_sentence) * 0.5 >= MIN_SPEECH_DURATION:
                sentence_audio = np.concatenate(current_sentence)
                current_sentence = []

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmpfile:
                    wav_write(tmpfile.name, SAMPLE_RATE, (sentence_audio * 32768).astype(np.int16))
                    with open(tmpfile.name, "rb") as f:
                        response = openai_client.audio.transcriptions.create(model="whisper-1", file=f, language=LANGUAGE)
                        final_text = response.text.strip()
                        await websocket.send_json({"type": "final", "text": final_text})

                    # 語者分離
                    result = diarization_pipeline(tmpfile.name)
                    speaker_info = []
                    for turn, _, speaker in result.itertracks(yield_label=True):
                        speaker_info.append({
                            "start": round(turn.start, 1),
                            "end": round(turn.end, 1),
                            "speaker": speaker
                        })
                    await websocket.send_json({"type": "speaker", "segments": speaker_info})
            else:
                current_sentence = []
