import os
import requests
from dotenv import load_dotenv

# === OpenAI API 金鑰 ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("⚠️ 找不到 OPENAI_API_KEY，請確認 .env 是否存在且正確")

def openai_whisper_transcribe(AUDIO_FILE_PATH):
    """
    使用 OpenAI Whisper API 進行語音辨識
    """
    if not os.path.exists(AUDIO_FILE_PATH):
        raise FileNotFoundError(f"音訊檔案不存在：{AUDIO_FILE_PATH}")

    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }
    files = {
        "file": (os.path.basename(AUDIO_FILE_PATH), open(AUDIO_FILE_PATH, "rb"), "audio/m4a"),
        "model": (None, "whisper-1"),
        "language": (None, "zh"),  # 或可指定如 "en", "zh"
    }

    response = requests.post(url, headers=headers, files=files)

    if response.status_code == 200:
        print("✅ 成功！語音辨識結果：")
        print(response.json()["text"])
        return response.json()["text"]
    else:
        print(f"❌ 發生錯誤！HTTP {response.status_code}")
        print(response.text)
        return None
if __name__ == "__main__":
    AUDIO_FILE_PATH = "./shorts.wav"
    response = openai_whisper_transcribe(AUDIO_FILE_PATH)
    # print(response)
    