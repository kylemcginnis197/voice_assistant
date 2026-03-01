import os
import tempfile
from datetime import datetime

import numpy as np
import soundfile as sf
import torch

# ---------------------------------------------------------------------------
# PyTorch 2.6 compatibility: weights_only now defaults to True, breaking
# fairseq / rvc_python checkpoints.
# ---------------------------------------------------------------------------
if not hasattr(torch, "_orig_load"):
    torch._orig_load = torch.load

def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return torch._orig_load(*args, **kwargs)

torch.load = _patched_torch_load

from kokoro import KPipeline

try:
    from rvc_python.infer import RVCInference
except ImportError:
    RVCInference = None

from config import (
    RECEIVE_SAMPLE_RATE,
    RVC_ENABLE,
    RVC_F0_METHOD,
    RVC_F0_UP_KEY,
    RVC_INDEX_PATH,
    RVC_INDEX_RATE,
    RVC_MODEL_PATH,
    RVC_PROTECT,
    TTS_SPEED,
    TTS_VOICE,
)
from log import get_logger

log = get_logger("speech")

OUTPUT_DIR = "speech_output"


def _select_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu:0"


class Speech:
    """
    Kokoro TTS → RVC voice conversion pipeline.

    All heavy models are loaded in __init__ so subsequent calls to speak()
    are as fast as possible. Each call to speak() saves a timestamped .wav
    file under speech_output/.
    """

    def __init__(
        self,
        model_path: str = RVC_MODEL_PATH,
        index_path: str = RVC_INDEX_PATH,
        voice: str = TTS_VOICE,
        speed: float = TTS_SPEED,
    ):
        self.voice = voice
        self.speed = speed
        self.device = _select_device()

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # --- Kokoro TTS ---
        try:
            self.pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
        except Exception as e:
            raise RuntimeError(f"[Speech] Kokoro init failed: {e}") from e

        # --- RVC engine ---
        if RVC_ENABLE:
            try:
                self.rvc = RVCInference(device=self.device)
                self.rvc.set_params(
                    f0method=RVC_F0_METHOD,
                    f0up_key=RVC_F0_UP_KEY,
                    index_rate=RVC_INDEX_RATE,
                    protect=RVC_PROTECT,
                    resample_sr=RECEIVE_SAMPLE_RATE,
                )
            except Exception as e:
                raise RuntimeError(f"[Speech] RVC init failed: {e}") from e

            self.load_rvc_model(model_path, index_path)

    def load_rvc_model(self, model_path: str, index_path: str = "") -> None:
        """Load (or hot-swap) the RVC model. Accepts absolute or relative paths."""
        model_path = os.path.abspath(model_path)
        index_path = os.path.abspath(index_path) if index_path else ""

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"[Speech] Model not found: {model_path}")
        if index_path and not os.path.isfile(index_path):
            raise FileNotFoundError(f"[Speech] Index not found: {index_path}")

        try:
            self.rvc.load_model(model_path, index_path=index_path)
        except Exception as e:
            raise RuntimeError(f"[Speech] Failed to load RVC model: {e}") from e

    def speak(self, text: str) -> str:
        """
        Convert text → Kokoro TTS → RVC voice conversion → timestamped .wav.

        Saves the file to speech_output/YYYY-MM-DD_HH-MM-SS.wav and returns
        the path. On RVC failure, saves raw TTS audio instead of crashing.
        """
        if not text or not text.strip():
            raise ValueError("[Speech] speak() called with empty text.")

        timestamp = datetime.now().strftime("model_output")
        out_path = os.path.join(OUTPUT_DIR, f"{timestamp}.wav")

        tts_audio = self._tts(text)

        if RVC_ENABLE:
            tts_audio = self._rvc(tts_audio)
            
        sf.write(out_path, tts_audio, RECEIVE_SAMPLE_RATE)
        return out_path

    def _tts(self, text: str) -> np.ndarray:
        chunks = []
        for _gs, _ps, chunk in self.pipeline(text=text, voice=self.voice, speed=self.speed):
            if chunk is not None and len(chunk):
                if isinstance(chunk, torch.Tensor):
                    chunk = chunk.cpu().numpy()
                chunks.append(np.asarray(chunk, dtype=np.float32))
        if not chunks:
            raise RuntimeError("[Speech] Kokoro returned no audio chunks.")
        return np.concatenate(chunks)

    def _rvc(self, audio: np.ndarray) -> np.ndarray:
        tmp_in = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        in_path, out_path = tmp_in.name, tmp_out.name
        tmp_in.close()
        tmp_out.close()
        try:
            sf.write(in_path, audio, 24_000)
            self.rvc.infer_file(input_path=in_path, output_path=out_path)
            result, _ = sf.read(out_path, dtype="float32")
            return result
        except Exception as e:
            log.warning(f"RVC conversion failed ({e}), returning raw TTS audio.")
            return audio
        finally:
            for path in (in_path, out_path):
                try:
                    os.remove(path)
                except OSError:
                    pass
