# Voice Assistant

A speech-to-speech AI voice assistant powered by Claude. It listens for a wake word, transcribes speech with Whisper, processes requests through Claude with tool integrations, and responds with synthesized speech via Kokoro TTS.

## Features

- **Wake Word Detection** - Hands-free activation using OpenWakeWord
- **Speech-to-Text** - OpenAI Whisper for accurate transcription
- **LLM Processing** - Claude (Sonnet/Opus) with tool use and adaptive thinking
- **Text-to-Speech** - Kokoro TTS with optional RVC voice conversion
- **Smart Home Control** - Govee light control (toggle, brightness, color)
- **Spotify Integration** - Search, play, pause, skip tracks
- **Weather** - Current conditions via WeatherAPI
- **Context Management** - Automatic token compaction for long conversations

## Project Structure

```
voice_assistant/
├── main.py              # Entry point
├── audio.py             # Audio I/O, wake word, VAD, transcription
├── model.py             # Claude API client & tool management
├── speech.py            # TTS (Kokoro + optional RVC)
├── config.py            # Configuration constants
├── log.py               # Logging setup
├── prompts/
│   └── system_prompt.md # System prompt & persona
├── tools/
│   ├── tools.py         # Tool registry
│   ├── weather.py       # Weather API integration
│   ├── spotify.py       # Spotify playback control
│   └── govee/
│       ├── controller.py # Govee device control
│       └── govee_lib.py  # Govee API wrapper
```

## Setup

### Prerequisites

- Python 3.10
- A microphone and speaker

### 1. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download OpenWakeWord models

The wake word ONNX models are not bundled with the package and must be downloaded separately. Run this once after installing:

```bash
python -c "import openwakeword; openwakeword.utils.download_models()"
```

### 4. Create your `.env` file

Copy the example and fill in your keys:

```bash
cp .env.example .env
```

Then edit `.env`:

```env
# Required
ANTHROPIC_API_KEY=your-anthropic-api-key

# Optional — app runs without these, tools just won't be available
GOVEE_API_KEY=your-govee-api-key
WEATHER_API=your-weatherapi-key
SPOTIPY_CLIENT_ID=your-spotify-client-id
SPOTIPY_CLIENT_SECRET=your-spotify-client-secret
SPOTIPY_REDIRECT_URI=http://localhost:8888/callback
```

### 5. Run

```bash
python main.py
```

---

## Windows-Specific Notes

### tflite-runtime is not available on Windows for Python 3.10

OpenWakeWord defaults to TFLite models which require `tflite-runtime`. This package has no Windows wheel for Python 3.10. The code is already patched to use the ONNX backend instead (`inference_framework="onnx"`), which works fine — no action needed.

### rvc-python requires Microsoft C++ Build Tools

RVC voice conversion (`rvc-python`) depends on `fairseq`, which must be compiled from source on Windows. This requires [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

RVC is **disabled by default** (`RVC_ENABLE = False` in `config.py`) and the import is handled gracefully, so the assistant runs fine without it. If you want to enable RVC:

1. Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Run `pip install rvc-python`
3. Set `RVC_ENABLE = True` in `config.py` and configure the model paths

---

## Configuration

Key settings in `config.py`:

| Setting | Default | Description |
|---|---|---|
| `AI_MODEL` | `sonnet 4.6` | Claude model to use |
| `TTS_VOICE` | `am_puck` | Kokoro voice ID |
| `TTS_SPEED` | `1.0` | Speech speed multiplier |
| `WAKE_WORD_THRESHOLD` | `0.5` | Wake word detection sensitivity |
| `VAD_SPEECH_THRESHOLD` | `0.65` | Voice activity detection sensitivity |
| `SILENCE_DURATION` | `0.5` | Seconds of silence before stopping recording |
| `CONVERSATION_TIMEOUT` | `10` | Seconds of silence before returning to wake word mode |
| `RVC_ENABLE` | `False` | Enable RVC voice conversion |

---

## Known Warnings (harmless)

These warnings appear on startup but do not affect functionality:

- `dropout option adds dropout after all but last recurrent layer` — Silero VAD model architecture quirk, no impact
- `torch.nn.utils.weight_norm is deprecated` — Kokoro uses an older PyTorch internal API, still works
- `FP16 is not supported on CPU; using FP32 instead` — Whisper runs in full precision on CPU. Transcription is slower but correct. Use a CUDA GPU to eliminate this

---

## How It Works

1. **Wake word** - Listens continuously until the activation phrase is detected
2. **Record** - Captures speech until 0.5s of silence (using Silero VAD)
3. **Transcribe** - Whisper converts audio to text
4. **Process** - Claude processes the request, optionally calling tools
5. **Speak** - Kokoro TTS synthesizes the response and plays it back
6. **Loop** - Returns to listening for the next command or wake word
