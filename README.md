# streaming_stt

## 模擬串流
```arduino
音訊 → 分段（200ms） 
     → VAD判斷 
        → 有聲 → buffer 
        → 無聲連續5段 → trigger 語音結束
              → Whisper 中文轉譯
              → pyannote.audio 分離語者
              → 每段語音獨立轉譯 + 輸出
```

```bash
pip install pydub webrtcvad soundfile numpy faster-whisper pyannote.audio
```

## 模擬語音串流 + 即時語音辨識（支援中文）+ 語者分離 + VAD + 即時修正
- 將一段完整音訊檔（例如 .wav）切成每 500ms 的區塊，模擬串流輸入 <br>
- 使用 VAD 判斷每一段是否為有效語音 <br>
- 對有效語音段執行語音辨識（Whisper） <br>
- 使用 pyannote 語者分離模型 <br>
- 遇到斷句時執行整段修正（即時修正句子） <br>
```bash
pip install faster-whisper webrtcvad pyannote-audio torch torchaudio
```

## 專案結構
```bash
realtime_transcriber/
├── app.py                  # WebSocket server
├── transcriber.py          # Whisper + Diarization
├── vad.py                  # VAD 工具
├── utils.py                # 音訊工具（切片、格式轉換）
├── config.py               # 模型與設定初始化
├── frontend/
│   ├── index.html
│   └── app.js
├── requirements.txt
└── .env
```
## 啟動方式
- 前端互動
```bash
# 安裝套件
pip install -r requirements.txt

# 執行 WebSocket 後端
uvicorn app:app --reload

# 執行前端
cd frondent
python -m http.server 5500
# 瀏覽器打開 http://localhost:5500
```
- 僅 console 模式
```bash
python run_local.py
```

## 一邊播放音檔、一邊顯示語音辨識字幕
```css
前端（HTML + JS）
│
├─ 播放音檔（audio tag）
└─ WebSocket：接收字幕更新、顯示在畫面上

後端（Flask + WebSocket）
└─ 邊播音檔、邊執行 transcriber.process_chunk(...)，透過 socket 傳字幕
```
### 安裝必要套件
```bash
pip install flask flask-socketio gevent soundfile torchaudio
```
### 使用方式
```bash
python app.py
```
- 然後瀏覽 http://localhost:5000/
- 當你按下「播放」，字幕會一邊顯示 [暫定]、再變成 [修正]！