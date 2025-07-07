import os
import webrtcvad
from openai import OpenAI
from dotenv import load_dotenv
from pyannote.audio import Pipeline

# Load secrets
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Audio settings
CHUNK_DURATION = 0.5
SAMPLE_RATE = 16000
VAD_MODE = 3
MIN_SPEECH_DURATION = 1.5
LANGUAGE = "zh"

# Init clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
vad = webrtcvad.Vad(VAD_MODE)
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HUGGINGFACE_TOKEN)
