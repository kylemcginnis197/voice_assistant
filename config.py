import pyaudio
import asyncio

# USER SETTINGS -------------------------------

# Seconds of silence after response before requiring the wake word to be said again.
CONVERSATION_TIMEOUT = 10  


# Seconds of continuous silence before returning
SILENCE_DURATION = 1.5

# Model Details
AI_MODEL = "claude-haiku-4-5-20251001"

# The maximum size of the context window before the model compacts everything (via summerization)
INPUT_TOKEN_LIMIT = 75_000
OUTPUT_TOKEN_LIMIT = 4_096

# Dynamically include tools in context window based on user query
TOOL_EMBEDDINGS_RAG = False
TOOL_RAG_TOP_K = 10

# Subagent supervisor
SUBAGENT_MAX_RETRIES = 2                            # how many times the supervisor can reject before giving up
SUBAGENT_SUPERVISOR_MODEL = "claude-sonnet-4-6"     # model used to review subagent output
SUBAGENT_MODEL = "claude-haiku-4-5-20251001"        # model subagent performs work with.

# ---------------------------------------------

# DEVELOPER SETTINGS --------------------------

# Audio ingestion settings and parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 4096

# Voice activity Detection
VAD_SPEECH_THRESHOLD = 0.65 # Silero probability above which a window counts as speech
VAD_WINDOW_SIZE = 512       # Minimum chunk size (samples) the model accepts at 16 kHz

# Open Wake Word
WAKE_WORD_THRESHOLD = 0.5   # confidence threshold to trigger assistant

# RVC Settings
RVC_ENABLE = False
RVC_MODEL_PATH = "KanyeWest808sandHeartBreakEra/KanyeWest808sandHeartBreakEra_375e_18750s.pth"
RVC_INDEX_PATH = "KanyeWest808sandHeartBreakEra/KanyeWest808sandHeartBreakEra.index"
RVC_F0_METHOD = "rmvpe" # pitch extraction method: harvest, crepe, rmvpe
RVC_F0_UP_KEY = -2      # pitch shift in semitones
RVC_INDEX_RATE = 0.81   # how much the index influences the timbre (0–1)
RVC_PROTECT = 0.34      # protect voiceless consonants (0–0.5)

# TTS (Kokoro)
TTS_VOICE = "am_puck"   # kokoro voice ID
TTS_SPEED = 1.3           # playback speed multiplier
# ---------------------------------------------

# Model state ---------------------------------
ASSISTANT_STATE = "WAITING"
ASSISTANT_QUEUE: asyncio.Queue[dict] = asyncio.Queue()
