from config import ASSISTANT_QUEUE, SUBAGENT_MAX_RETRIES, SUBAGENT_SUPERVISOR_MODEL, SUBAGENT_MODEL, INPUT_TOKEN_LIMIT, OUTPUT_TOKEN_LIMIT
from model import Model
from pydantic import BaseModel, Field
from datetime import timedelta
from typing import Optional
from log import get_logger
log = get_logger("main")
import anthropic
import asyncio
import time

# Import Tools
from . import weather
from .govee.controller import govee_controller
from . import spotify
from . import radarr
from . import sonarr

# Weather API
TOOLS = [
    weather.call_weather_api
]

# Govee
if govee_controller:
    TOOLS.extend([
        govee_controller.set_brightness,
        govee_controller.set_color,
        govee_controller.toggle_lights
    ])

if spotify.sp:
    TOOLS.extend([
        spotify.sp.get_recently_played_songs,
        spotify.sp.start_playback,
        spotify.sp.pause_playback,
        spotify.sp.next_track,
        spotify.sp.previous_track,
        spotify.sp.search,
    ])

# Radarr
if radarr.radarr:
    TOOLS.extend([
        radarr.radarr.search_movie,
        radarr.radarr.add_movie,
        radarr.radarr.list_movies,
        radarr.radarr.check_queue,
        radarr.radarr.disk_space,
    ])

# Sonarr
if sonarr.sonarr:
    TOOLS.extend([
        sonarr.sonarr.search_series,
        sonarr.sonarr.add_series,
        sonarr.sonarr.list_series,
        sonarr.sonarr.search_season,
        sonarr.sonarr.search_episode
    ])

class EndCoversation(BaseModel):
    reason: str = Field(description="Describe why you believe the user no longer requires your active listening. Regardless of the reason, calling this tool will require the user to say the wake word before getting your attention again.")

def _end_conversation(args: EndCoversation):
    """End the active conversation and return to wake-word listening. Call this when the user's request is fulfilled or they are done talking."""
    import config
    config.ASSISTANT_STATE = "WAITING"
    log.info(f"Model ended conversation, reason: {args.reason}")

class ScheduleTask(BaseModel):
    hours: int = Field(description="Hours until task should occur")
    minutes: int = Field(description="Minutes until task should occur")
    seconds: int = Field(description="Seconds until task should occur")
    model_prompt: Optional[str] = Field(default=None, description="Prompt to pass to the model so you can access the internet or tools as needed to complete task. If no model prompt is provided, be sure to provide a text output for the TTS model to tell the user.")
    tts_text: Optional[str] = Field(default=None, description="Provide text that will be spoken to the user. Useful let the user know a timer is done or for reminders. If no text is provided, be sure to provide a task for the model prompt.")

async def schedule_task_tool(delay_seconds: float, payload: dict): 
    await asyncio.sleep(delay_seconds)
    await ASSISTANT_QUEUE.put(item=payload)
    log.info(f"Scheduled task fired: {payload}")

def _schedule_task(args: ScheduleTask):
    """Schedules a task that will occur after a set amount of time"""
    hours = args.hours
    minutes = args.minutes
    seconds = args.seconds
     
    # convert time to total seconds
    total_seconds = timedelta(hours=hours, minutes=minutes, seconds=seconds).total_seconds()

    if args.model_prompt or args.tts_text:
        asyncio.create_task(schedule_task_tool(delay_seconds=total_seconds, payload={
            "prompt": args.model_prompt,
            "tts_text": args.tts_text
        }))
    else:
        return "Failed to schedule task, Pass an input for either model_prompt or tts_text"

_supervisor_client = anthropic.AsyncAnthropic()

async def run_supervisor(task_description: str, context: list[str], result: str) -> tuple[bool, str]:
    """Reviews subagent output. Returns (approved, feedback_if_rejected)."""
    response = await _supervisor_client.messages.create(
        model=SUBAGENT_SUPERVISOR_MODEL,
        max_tokens=256,
        system=(
            "You are a quality assurance supervisor for an AI assistant. "
            "Evaluate whether the subagent fully and correctly completed the task. "
            "Reply with exactly one of:\n"
            "APPROVED\n"
            "REJECTED: <brief explanation of what is missing or incorrect>"
        ),
        messages=context + [{
            "role": "user",
            "content": f"Everything previously mention is from the subagent, their original task: {task_description}\n\n and subagent result:\n{result}"
        }]
    )
    text = response.content[0].text.strip()
    if text.upper().startswith("APPROVED"):
        return True, ""
    return False, text.removeprefix("REJECTED:").strip()

async def run_subagent(task_description: str, name: str = None):
    """Spawns a subagent to complete a task, with supervisor review and retries."""
    t0 = time.monotonic()
    log.info(f"[subagent] Starting task: {task_description}")

    try:
        subagent_client = Model(tools=TOOLS, always_included_tools=[], name="subagent" if name is None else name, web_search=True)
        subagent_client.set_model(SUBAGENT_MODEL)
        subagent_client.set_input_tokens(INPUT_TOKEN_LIMIT)
        subagent_client.set_output_tokens(OUTPUT_TOKEN_LIMIT)

        feedback = ""
        result = None
        total_attempts = SUBAGENT_MAX_RETRIES + 1

        for attempt in range(total_attempts):
            if attempt == 0:
                attempt_prompt = f"You are a subagent tasked to perform the following: {task_description}"
            else:
                attempt_prompt = (
                    f"You are a subagent tasked to perform the following: {task_description}\n\n"
                    f"Your previous attempt was reviewed and rejected. Supervisor feedback: {feedback}\n\n"
                    f"Please address the feedback and redo the task properly."
                )

            result = await subagent_client.call_model(input=attempt_prompt) or "No response generated."

            approved, feedback = await run_supervisor(task_description, subagent_client.context_window, result)
            log.info(f"[subagent] Attempt {attempt + 1}/{total_attempts}: {'approved' if approved else f'rejected — {feedback}'}")

            if approved or attempt == SUBAGENT_MAX_RETRIES:
                break

        log.info(f"[subagent] Task finished ({time.monotonic() - t0:.2f}s)")
        prompt = f"Subagent task completed.\n\nOriginal task: {task_description}\n\nResult: {result}"
    except Exception as e:
        log.error(f"[subagent] Task failed ({time.monotonic() - t0:.2f}s): {e}")
        prompt = f"Subagent task failed.\n\nOriginal task: {task_description}\n\nError: {e}"

    await ASSISTANT_QUEUE.put({"prompt": prompt, "tts_text": None})

class SubAgent(BaseModel):
    name: str = Field(description="Provide a name related to the task the subagent is performing.")
    task_description: str = Field(description="Describe the task that your subagent needs to perform.")

# Still a work in progress...
def _start_subagent(args: SubAgent):
    """Deploy a background subagent to perform a task autonomously. The result will be delivered back when complete."""
    asyncio.create_task(run_subagent(task_description=args.task_description, name=args.name))
    return f"Subagent deployed. Task is being carried out in the background!"