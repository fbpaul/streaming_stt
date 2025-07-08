import numpy as np
import tempfile
import time
import json
from scipy.io.wavfile import write as wav_write
from config import SAMPLE_RATE, CHUNK_DURATION, MIN_SPEECH_DURATION, LANGUAGE, openai_client, diarization_pipeline
from vad import is_speech


class SpeakerMapper:
    def __init__(self):
        self.mapping = {}
        self.counter = 1

    def get(self, raw_name):
        if raw_name not in self.mapping:
            self.mapping[raw_name] = f"Speaker {self.counter}"
            self.counter += 1
        return self.mapping[raw_name]


class StreamingTranscriber:
    def __init__(self, vad):
        self.vad = vad
        self.buffer = []
        self.audio_cache = []  # 所有片段累積
        self.current_time = 0.0
        self.speaker_mapper = SpeakerMapper()
        self.last_speaker = None
        self.last_block = None
        self.last_end_time = 0.0
        self.output_path = "transcript.jsonl"
        self.interim_sentence = ""
        self.interim_id = 0
        self.max_interim_chunk = 6

        with open(self.output_path, "w", encoding="utf-8") as f:
            pass

    def process_chunk(self, chunk, websocket=None):
        self.audio_cache.append(chunk)
        result_to_return = None

        if is_speech(chunk, self.vad, SAMPLE_RATE):
            self.buffer.append(chunk)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmpfile:
                wav_write(tmpfile.name, SAMPLE_RATE, (chunk * 32768).astype(np.int16))
                with open(tmpfile.name, "rb") as f:
                    response = openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f,
                        language=LANGUAGE
                    )
                    text = response.text.strip()

                    # 串接暫定文字
                    if text:
                        self.interim_sentence += text + " "
                        self.interim_id += 1

                        result_to_return = {
                            "type": "interim",
                            "id": self.interim_id,
                            "text": self.interim_sentence.strip()
                        }

        # 不再是語音段，或達到最長暫定句限制就觸發完整辨識
        if (not is_speech(chunk, self.vad, SAMPLE_RATE) and len(self.buffer) * CHUNK_DURATION >= MIN_SPEECH_DURATION) or \
           (len(self.buffer) >= self.max_interim_chunk):

            audio_segment = np.concatenate(self.buffer)
            self.buffer = []

            duration = len(audio_segment) / SAMPLE_RATE
            start_time = round(self.last_end_time, 2)
            end_time = round(self.last_end_time + duration, 2)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmpfile:
                wav_write(tmpfile.name, SAMPLE_RATE, (audio_segment * 32768).astype(np.int16))

                with open(tmpfile.name, "rb") as f:
                    response = openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f,
                        language=LANGUAGE
                    )
                    final_text = response.text.strip()

                result = diarization_pipeline(tmpfile.name)
                speaker_time_map = {}
                for turn, _, spk in result.itertracks(yield_label=True):
                    overlap_start = max(0, turn.start)
                    overlap_end = min(duration, turn.end)
                    overlap = max(0, overlap_end - overlap_start)
                    if overlap > 0:
                        speaker_time_map[spk] = speaker_time_map.get(spk, 0) + overlap

                raw_speaker = max(speaker_time_map.items(), key=lambda x: x[1])[0] if speaker_time_map else "Unknown"
                speaker = self.speaker_mapper.get(raw_speaker)

            final_result = {
                "type": "final",
                "speaker": speaker,
                "start": start_time,
                "end": end_time,
                "text": final_text
            }

            self.last_end_time = end_time

            if self.last_speaker == speaker and self.last_block:
                self.last_block["text"] += " " + final_text
                self.last_block["end"] = end_time
            else:
                self.last_block = final_result
                self.last_speaker = speaker
                with open(self.output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(final_result, ensure_ascii=False) + "\n")

            # 清除暫定累積
            self.interim_sentence = ""
            self.interim_id = 0
            result_to_return = final_result

        # ✅ 推進時間軸
        self.current_time += CHUNK_DURATION
        return result_to_return
