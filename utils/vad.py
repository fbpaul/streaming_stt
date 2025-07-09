import numpy as np

def float32_to_pcm16(audio_float):
    """將 float32 音訊轉為 PCM16 格式"""
    audio_float = np.clip(audio_float, -1, 1)
    return (audio_float * 32767).astype(np.int16).tobytes()

def split_audio_to_chunks(audio, sample_rate=16000, chunk_duration=0.5):
    """將音訊切成固定長度的片段"""
    chunk_size = int(chunk_duration * sample_rate)
    return [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]

def is_speech(chunk, vad, sample_rate=16000, threshold=0.5):
    """
    判斷該 chunk 是否為語音。
    若 chunk 中語音 frame 比例超過 threshold，回傳 True。
    """
    if not isinstance(chunk, np.ndarray) or chunk.dtype != np.float32:
        raise ValueError("輸入的 chunk 必須是 float32 numpy array。")

    pcm_data = float32_to_pcm16(chunk)

    frame_duration_ms = 30  # 每一小段ms長度
    frame_size = int(sample_rate * frame_duration_ms / 1000)
    byte_size = frame_size * 2  # 16-bit = 2 bytes

    num_frames = len(pcm_data) // byte_size
    if num_frames == 0:
        return False

    speech_frames = 0
    for i in range(num_frames):
        start = i * byte_size
        end = start + byte_size
        frame = pcm_data[start:end]
        if len(frame) < byte_size:
            continue
        try:
            if vad.is_speech(frame, sample_rate):
                speech_frames += 1
        except:
            continue

    speech_ratio = speech_frames / num_frames
    return speech_ratio >= threshold
