import numpy as np
import tempfile
import time
import json
import io
from scipy.io.wavfile import write as wav_write
import soundfile as sf
import torch
from faster_whisper import WhisperModel

from configs.config import SAMPLE_RATE, CHUNK_DURATION, MIN_SPEECH_DURATION, LANGUAGE, faster_whisper_model, diarization_pipeline
from utils.vad import is_speech


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
        # self.speaker_mapper = SpeakerMapper()
        self.last_speaker = None
        self.last_block = None
        self.last_end_time = 0.0
        self.silence_duration = 0.0 # 靜音時間
        self.output_path = "transcript.jsonl"
        self.interim_sentence = ""
        self.interim_id = 0
        self.max_interim_chunk = 6
        self.model = faster_whisper_model
        self.diarization_pipeline = diarization_pipeline

        with open(self.output_path, "w", encoding="utf-8") as f:
            pass
    
    @staticmethod
    def audio_to_bytes(audio_data, sample_rate):
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        return buffer


    def process_chunk(self, chunk, websocket=None):
        speaker = "Speaker 1"  # 預設說話者
        # total_start_time = time.perf_counter()

        self.audio_cache.append(chunk)
        result_to_return = None

        # is_speech_start = time.perf_counter()
        is_speech_result = is_speech(chunk, self.vad, SAMPLE_RATE)
        # is_speech_end = time.perf_counter()
        # print(f"🔍 判斷語音段耗時: {is_speech_end - is_speech_start:.2f}s")

        if is_speech_result:
            self.buffer.append(chunk)
            self.silence_duration = 0.0

            # audio_to_bytes_start = time.perf_counter()
            audio_bytes = self.audio_to_bytes(chunk, SAMPLE_RATE)
            # audio_to_bytes_end = time.perf_counter()
            # print(f"📦 音訊轉換為 bytes 耗時: {audio_to_bytes_end-audio_to_bytes_start:.2f}s")

            # transcribe_start = time.perf_counter()
            segments, _ = self.model.transcribe(
                audio_bytes,
                vad_filter=False,
                # vad_parameters=dict(min_silence_duration_ms=500),
                language=LANGUAGE,
                word_timestamps=True
            )
            # transcribe_end = time.perf_counter()
            # print(f"🧠 Interim Transcribe 耗時: {transcribe_end - transcribe_start:.2f}s")

            all_words = [word for segment in segments for word in segment.words]
            text = ' '.join([word.word for word in all_words])

            if text:
                self.interim_sentence += text + " "
                self.interim_id += 1
                result_to_return = {
                    "type": "interim",
                    "id": self.interim_id,
                    "text": self.interim_sentence.strip()
                }

        else:
            self.silence_duration += CHUNK_DURATION
            if self.silence_duration >= 1 and len(self.buffer) * CHUNK_DURATION >= MIN_SPEECH_DURATION:

                # concat_start = time.perf_counter()
                audio_segment = np.concatenate(self.buffer)
                # concat_end = time.perf_counter()
                # print(f"🔗 音訊拼接耗時: {concat_end - concat_start:.2f}s")

                self.buffer = []

                duration = len(audio_segment) / SAMPLE_RATE
                start_time = round(self.last_end_time, 2)
                end_time = round(self.last_end_time + duration, 2)

                # audio_to_bytes_start = time.perf_counter()
                audio_bytes = self.audio_to_bytes(audio_segment, SAMPLE_RATE)
                # audio_to_bytes_end = time.perf_counter()
                # print(f"📦 音訊轉換為 bytes 耗時: {audio_to_bytes_end-audio_to_bytes_start:.2f}s")

                # final_transcribe_start = time.perf_counter()
                segments, _ = self.model.transcribe(
                    audio_bytes,
                    vad_filter=False,
                    # vad_parameters=dict(min_silence_duration_ms=500),
                    language=LANGUAGE,
                    word_timestamps=True
                )
                # final_transcribe_end = time.perf_counter()
                # print(f"🧠 Final Transcribe 耗時: {final_transcribe_end - final_transcribe_start:.2f}s")

                all_words = [word for segment in segments for word in segment.words]
                final_text = ' '.join([word.word for word in all_words])

                if all_words:
                    start_time = round(self.last_end_time + all_words[0].start, 2)
                    end_time = round(self.last_end_time + all_words[-1].end, 2)
                else:
                    start_time = self.last_end_time
                    end_time = self.last_end_time + duration
                
                # Diarization 待補

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

                    # write_json_start = time.perf_counter()
                    with open(self.output_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(final_result, ensure_ascii=False) + "\n")
                    # write_json_end = time.perf_counter()
                    # print(f"📝 寫入 transcript.jsonl 耗時: {write_json_end - write_json_start:.2f}s")

                self.interim_sentence = ""
                self.interim_id = 0
                result_to_return = final_result

        self.current_time += CHUNK_DURATION
        # total_end_time = time.perf_counter()
        # print(f"⏱️ 整體 process_chunk 耗時: {total_end_time - total_start_time:.2f}s")

        return result_to_return

