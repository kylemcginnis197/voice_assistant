from . import weather
from .govee.controller import govee_controller
from . import spotify

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
        spotify.sp.start_playback,
        spotify.sp.pause_playback,
        spotify.sp.next_track,
        spotify.sp.previous_track,
        spotify.sp.search,
    ])