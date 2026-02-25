import asyncio
import time
_T_MAIN_START = time.monotonic()

from log import setup_logging, get_logger
setup_logging()
log = get_logger("main")
log.info(f"[STARTUP] main.py started")

from dotenv import load_dotenv
load_dotenv()

import pyaudio
import audio as audio_module

from pydantic import BaseModel, Field
from model import Model
from speech import Speech
from tools.tools import TOOLS
from audio import wait_for_wake_word, flush_queues, open_streams, wait_for_speech_start, wait_for_speech_end, mic_buffer, transcribe_audio, play_wav_file
from config import AI_MODEL, INPUT_TOKEN_LIMIT, OUTPUT_TOKEN_LIMIT, CONVERSATION_TIMEOUT


# End Conversation Tool
class EndCoversation(BaseModel):
    reason: str = Field(description="Describe why you believe the user no longer requires your active listening. Regardless of the reason, calling this tool will require the user to say the wake word before getting your attention again.")

def end_conversation(args: EndCoversation):
    """End the active conversation and return to wake-word listening. Call this when the user's request is fulfilled or they are done talking."""
    global model_state
    model_state = "WAITING"
    log.log(msg=f"Model ended conversation, reason: {args.reason}")

client = Model(tools=TOOLS, always_included_tools=[end_conversation], web_search=True)
client.set_model(AI_MODEL)
client.set_input_tokens(INPUT_TOKEN_LIMIT)
client.set_output_tokens(OUTPUT_TOKEN_LIMIT)

pya = pyaudio.PyAudio()

async def run():
    global model_state
    t0 = _T_MAIN_START
    models_ready = False

    mic_stream, speaker_stream = open_streams(pya)
    mic_stream.start_stream()
    speaker_stream.start_stream()
    log.info(f"[STARTUP] Streams open ({time.monotonic()-t0:.2f}s)")

    # Load all heavy models in parallel background threads
    oww_task     = asyncio.create_task(asyncio.to_thread(audio_module.create_wake_word_model))
    whisper_task = asyncio.create_task(asyncio.to_thread(audio_module.init_whisper_and_vad))
    speech_task  = asyncio.create_task(asyncio.to_thread(Speech))

    # Wait only for OpenWakeWord, then start listening immediately
    oww_model = await oww_task
    log.info(f"[STARTUP] Wake word model ready ({time.monotonic()-t0:.2f}s) — LISTENING")

    try:
        while True:
            flush_queues()
            mic_buffer.clear()

            # Ensure transcription + speech models are ready before use
            if not models_ready:
                await whisper_task
                speech_obj = await speech_task
                log.info(f"[STARTUP] All models ready ({time.monotonic()-t0:.2f}s)")
                models_ready = True

            await wait_for_wake_word(oww_model)

            flush_queues()
            mic_buffer.clear()

            log.info(f"Listening...")
            model_state = "LISTENING"

            while model_state == "LISTENING":
                try:
                    await asyncio.wait_for(wait_for_speech_start(), timeout=CONVERSATION_TIMEOUT)
                except asyncio.TimeoutError:
                    log.info(f"No speech detected for {CONVERSATION_TIMEOUT}s, returning to WAITING")
                    model_state = "WAITING"
                    break

                await wait_for_speech_end()

                ts = time.monotonic()

                audio_snapshot = bytes(mic_buffer)

                mic_buffer.clear()
                flush_queues()

                text = await transcribe_audio(audio_snapshot)
                log.info(f"Transcribed ({(time.monotonic() - ts):.2}s) {text}")

                response = await client.call_model(text, ts)
                log.info(f"Model response ({(time.monotonic() - ts):.2}s): {response}")

                if response and model_state == "LISTENING":
                    wav_path = await asyncio.to_thread(speech_obj.speak, response)
                    await play_wav_file(wav_path)

                flush_queues()
                mic_buffer.clear()

            mic_stream.stop_stream()
            mic_stream.start_stream()
    except asyncio.CancelledError:
        pass
    finally:
        mic_stream.stop_stream()
        mic_stream.close()
        speaker_stream.stop_stream()
        speaker_stream.close()
        pya.terminate()
        log.info("Connection closed.")


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
