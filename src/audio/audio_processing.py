import sounddevice as sd
import numpy as np

# Global audio state
audio_level = 0.0
audio_history = []
HISTORY_SIZE = 10

def audio_callback(indata, frames, time, status):
    global audio_level
    
    # Volume (RMS energy)
    volume_norm = np.linalg.norm(indata) / len(indata)
    audio_level = volume_norm

    global audio_history
    audio_history.append(volume_norm)
    if len(audio_history) > HISTORY_SIZE:
        audio_history.pop(0)


def start_audio_stream():
    stream = sd.InputStream(callback=audio_callback)
    stream.start()
    return stream


def get_audio_state():
    global audio_history

    if len(audio_history) == 0:
        return "Silent / Low Energy"

    avg_level = sum(audio_history) / len(audio_history)

    if avg_level > 0.02:
        return "Speaking (High Energy)"
    elif avg_level > 0.005:
        return "Speaking (Normal)"
    else:
        return "Silent / Low Energy"