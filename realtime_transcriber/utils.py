import numpy as np

def float32_to_pcm16(audio_float):
    audio_float = np.clip(audio_float, -1, 1)
    return (audio_float * 32767).astype(np.int16).tobytes()

def is_speech(chunk, vad, sample_rate, threshold=0.5):
    pcm = float32_to_pcm16(chunk)
    frame_duration = 30
    frame_size = int(sample_rate * frame_duration / 1000)
    byte_size = frame_size * 2

    num_frames = len(pcm) // byte_size
    if num_frames == 0:
        return False

    speech_count = 0
    for i in range(num_frames):
        frame = pcm[i * byte_size:(i + 1) * byte_size]
        if len(frame) < byte_size:
            continue
        if vad.is_speech(frame, sample_rate):
            speech_count += 1

    return (speech_count / num_frames) >= threshold
