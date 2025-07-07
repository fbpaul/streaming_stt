import os
from dotenv import load_dotenv
import webrtcvad
from openai import OpenAI
from pyannote.audio import Pipeline

# 讀取 .env 環境變數
load_dotenv()

# 語音處理參數
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5  # 每段 0.5 秒
MIN_SPEECH_DURATION = 1.5  # 語音段必須大於此長度才送出辨識
VAD_MODE = 3  # VAD 模式：3 為最嚴格
LANGUAGE = "zh"

# 初始化 OpenAI Whisper 客戶端
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# 初始化 pyannote 語者分離模型
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HF_TOKEN)

# 初始化 WebRTC VAD
vad = webrtcvad.Vad(VAD_MODE)
