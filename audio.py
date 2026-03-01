import asyncio
import queue as thread_queue
import time

import numpy as np
import pyaudio
import soundfile as sf
import torch
from openwakeword.model import Model
from silero_vad import load_silero_vad

import whisper
import numpy as np

from config import (
    CHANNELS,
    CHUNK_SIZE,
    FORMAT,
    RECEIVE_SAMPLE_RATE,
    SEND_SAMPLE_RATE,
    WAKE_WORD_THRESHOLD,
    VAD_SPEECH_THRESHOLD,
    VAD_WINDOW_SIZE,
    SILENCE_DURATION
)
from log import get_logger

log = get_logger("audio")

# Thread-safe queues bridging PyAudio callback threads <-> asyncio tasks
mic_queue = thread_queue.Queue(maxsize=50)
speaker_queue = thread_queue.Queue()  # unbounded — callback drains continuously
vad_queue = thread_queue.Queue(maxsize=50)  # every mic chunk, for speech detection

# Speaker callback internal buffer (only accessed from the audio output thread)
_speaker_buf = bytearray()
mic_buffer = bytearray()

# ---------------------------------------------------------------------------
# PyAudio callbacks (run on high-priority audio threads)
# ---------------------------------------------------------------------------
talking_time = None
rms_historic = []

def get_volume(audio_float32):
    rms = np.sqrt(np.mean(np.square(audio_float32)))
    
    if rms > 0:
        db = 20 * np.log10(rms)
    else:
        db = -90 # Silence
        
    return rms, db

def mic_callback(in_data, frame_count, time_info, status):
    global rms_historic

    audio_int16 = np.frombuffer(in_data, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0

    rms = get_volume(audio_float32)
    rms_historic.append(rms[0])

    while len(rms_historic) > 20:
        rms_historic.pop(0)

    try:
        vad_queue.put_nowait(in_data)
    except thread_queue.Full:
        pass

    mic_buffer.extend(in_data)

    try:
        mic_queue.put_nowait(in_data)
    except thread_queue.Full:
        pass 

    return (None, pyaudio.paContinue)

def speaker_callback(in_data, frame_count, time_info, status):
    needed = frame_count * CHANNELS * 2  # 2 bytes per int16 sample

    while len(_speaker_buf) < needed:
        try:
            _speaker_buf.extend(speaker_queue.get_nowait())
        except thread_queue.Empty:
            break

    if len(_speaker_buf) >= needed:
        out = bytes(_speaker_buf[:needed])
        del _speaker_buf[:needed]
    else:
        out = bytes(_speaker_buf) + b"\x00" * (needed - len(_speaker_buf))
        _speaker_buf.clear()

    return (out, pyaudio.paContinue)


# ---------------------------------------------------------------------------
# Wake word detection
# ---------------------------------------------------------------------------

async def wait_for_wake_word(oww):
    """Blocks until a wake word is detected via mic_queue."""
    oww.reset()
    log.info("Listening for wake word...")

    while True:
        data = await asyncio.to_thread(mic_queue.get)
        audio = np.frombuffer(data, dtype=np.int16)
        prediction = oww.predict(audio)
        for name, score in prediction.items():
            if score > WAKE_WORD_THRESHOLD:
                return name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def flush_queues():
    """Drain all queues and clear the speaker buffer."""
    for q in (mic_queue, speaker_queue, vad_queue):
        while not q.empty():
            try:
                q.get_nowait()
            except thread_queue.Empty:
                break
    _speaker_buf.clear()


def open_streams(pya: pyaudio.PyAudio):
    """Create and return (mic_stream, speaker_stream) with callbacks attached."""
    mic_info = pya.get_default_input_device_info()

    mic_stream = pya.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SEND_SAMPLE_RATE,
        input=True,
        input_device_index=mic_info["index"],
        frames_per_buffer=CHUNK_SIZE,
        stream_callback=mic_callback,
    )

    speaker_stream = pya.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RECEIVE_SAMPLE_RATE,
        output=True,
        frames_per_buffer=CHUNK_SIZE,
        stream_callback=speaker_callback,
    )

    return mic_stream, speaker_stream

audio_model = None
vad_model = None


def init_whisper_and_vad():
    """Load Whisper + Silero VAD models. Intended to be called in a background thread."""
    global audio_model, vad_model
    audio_model = whisper.load_model("turbo")
    vad_model = load_silero_vad()


def create_wake_word_model():
    """Instantiate and return an OpenWakeWord Model. Intended to be called in a background thread."""
    oww = Model(inference_framework="onnx")
    oww.reset()
    return oww


def _vad_max_prob(chunk: bytes) -> float:
    """Score a raw audio chunk with Silero VAD and return the peak probability."""
    audio_int16 = np.frombuffer(chunk, dtype=np.int16)
    audio_f32 = torch.from_numpy(audio_int16.astype(np.float32) / 32768.0)
    max_prob = 0.0
    for i in range(0, len(audio_f32) - VAD_WINDOW_SIZE + 1, VAD_WINDOW_SIZE):
        prob = vad_model(audio_f32[i : i + VAD_WINDOW_SIZE], SEND_SAMPLE_RATE).item()
        max_prob = max(max_prob, prob)
    return max_prob


async def wait_for_speech_start() -> None:
    """Returns once speech is first detected. Resets VAD state."""
    vad_model.reset_states()
    while True:
        chunk = await asyncio.to_thread(vad_queue.get)
        if _vad_max_prob(chunk) >= VAD_SPEECH_THRESHOLD:
            return


async def wait_for_speech_end() -> bool:
    """Returns True once the user stops speaking for SILENCE_DURATION seconds.

    Assumes speech has already started (call wait_for_speech_start first).
    """
    last_speech_time = time.monotonic()

    while True:
        chunk = await asyncio.to_thread(vad_queue.get)
        max_prob = _vad_max_prob(chunk)

        if max_prob >= VAD_SPEECH_THRESHOLD:
            last_speech_time = time.monotonic()
        elif time.monotonic() - last_speech_time >= SILENCE_DURATION:
            return True


async def transcribe_audio(data: bytes) -> str:
    """Transcribes audio data using the Whisper model and returns the text."""
    try:
        audio_int16 = np.frombuffer(data, dtype=np.int16)
        auto_float32 = audio_int16.astype(np.float32) / 32768.0
        result = audio_model.transcribe(audio = auto_float32)
        return result["text"].strip()
    except Exception as e:
        log.error(f"Transcription exception: {e}")
        return ""


async def play_wav_file(path: str) -> None:
    """Stream a .wav file through the speaker_queue and wait for playback to finish."""
    audio_f32, sr = sf.read(path, dtype="float32")
    audio_int16 = (audio_f32 * 32767).astype(np.int16)
    raw = audio_int16.tobytes()

    chunk_bytes = CHUNK_SIZE * 2  # CHUNK_SIZE samples × 2 bytes per int16 sample
    for i in range(0, len(raw), chunk_bytes):
        speaker_queue.put(raw[i : i + chunk_bytes])

    duration = len(audio_int16) / sr
    await asyncio.sleep(duration + 0.3)  # small buffer to let the last chunk drain
