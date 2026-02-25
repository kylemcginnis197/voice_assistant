import pyaudio

# Audio
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 4096

VAD_SPEECH_THRESHOLD = 0.65   # Silero probability above which a window counts as speech
VAD_WINDOW_SIZE = 512        # Minimum chunk size (samples) the model accepts at 16 kHz
SILENCE_DURATION = 0.5       # Seconds of continuous silence before returning

# Wake word
WAKE_WORD_THRESHOLD = 0.5

# Conversation
CONVERSATION_TIMEOUT = 10  # Seconds of silence after response before returning to WAITING

# Model
AI_MODEL = "sonnet 4.6"
INPUT_TOKEN_LIMIT = 75_000
OUTPUT_TOKEN_LIMIT = 4_096

# Tool Embeddings Rag Model
TOOL_EMBEDDINGS_RAG = True
TOOL_RAG_TOP_K = 3

# RVC Voice Conversion
RVC_ENABLE = False
RVC_MODEL_PATH = "KanyeWest808sandHeartBreakEra/KanyeWest808sandHeartBreakEra_375e_18750s.pth"
RVC_INDEX_PATH = "KanyeWest808sandHeartBreakEra/KanyeWest808sandHeartBreakEra.index"
RVC_F0_METHOD = "rmvpe"   # pitch extraction method: harvest, crepe, rmvpe
RVC_F0_UP_KEY = -2           # pitch shift in semitones
RVC_INDEX_RATE = 0.81        # how much the index influences the timbre (0–1)
RVC_PROTECT = 0.34          # protect voiceless consonants (0–0.5)

# TTS (Kokoro)
TTS_VOICE = "am_puck"      # kokoro voice ID
TTS_SPEED = 1             # playback speed multiplier
