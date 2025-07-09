import os
from dotenv import load_dotenv
import webrtcvad
from openai import OpenAI
from faster_whisper import WhisperModel
import torch
from pyannote.audio import Pipeline

# 讀取 .env 環境變數
load_dotenv()

# 語音處理參數
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5  # 每段秒數
MIN_SPEECH_DURATION = 6  # 語音段若大於此長度就一定會送出辨識，加快辨識感
VAD_MODE = 3  # VAD 模式：3 為最嚴格
LANGUAGE = "zh"
FILE = "static/shorts.wav"
MODEL_SIZE = "medium"  # faster-whisper 模型大小
DIARIZATION_MODEL = "pyannote/speaker-diarization"  # pyannote 語者分離模型

# 初始化 OpenAI Whisper 客戶端
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# 初始化faster_whisper 模型
faster_whisper_model = WhisperModel(MODEL_SIZE, device="cuda" if torch.cuda.is_available() else "cpu", compute_type = "auto")

# 初始化 pyannote 語者分離模型
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
diarization_pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=HF_TOKEN)

# 初始化 WebRTC VAD
vad = webrtcvad.Vad(VAD_MODE)
