from pydantic import BaseModel, Field
from typing import Optional, List
from . import govee_lib

ROOM_LIGHTS = [
    "Overhead 1",
    "Overhead 2",
    "Ambient light"
]

DOWNSTAIRS_LIGHTS = [
    "Cone"
]

class ToggleLights(BaseModel):
    action: str = Field(description="Tell the Govee Controller to turn the lights 'on' or 'off'")
    room: Optional[str] = Field(default=None, description="Room to target: 'bedroom' or 'downstairs'. Use this OR device_names, not both.")
    device_names: Optional[List[str]] = Field(default=None, description="List of specific light names to target, e.g. ['Overhead 1', 'Cone']. Use this OR room.")

class SetBrightness(BaseModel):
    brightness: int = Field(description="Brightness level from 0 to 100")
    room: Optional[str] = Field(default=None, description="Room to target: 'bedroom' or 'downstairs'. Use this OR device_names, not both.")
    device_names: Optional[List[str]] = Field(default=None, description="List of specific light names to target, e.g. ['Overhead 1', 'Ambient light']. Use this OR room.")

class SetColor(BaseModel):
    color: str = Field(description="Color name: red, green, blue, white, yellow, cyan, magenta, purple, orange, pink, warm, cool")
    room: Optional[str] = Field(default=None, description="Room to target: 'bedroom' or 'downstairs'. Use this OR device_names, not both.")
    device_names: Optional[List[str]] = Field(default=None, description="List of specific light names to target, e.g. ['Overhead 1', 'Cone']. Use this OR room.")

class Govee:
    def __init__(self):
        self.API = govee_lib.get_api_key()
        self.devices = govee_lib.list_devices()

        for device in self.devices:
            device["room"] = "bedroom" if device.get("deviceName") in ROOM_LIGHTS else "downstairs"

    def _filter_devices(self, room: Optional[str], device_names: Optional[List[str]]):
        if room:
            return [d for d in self.devices if d["room"] == room.lower()]
        elif device_names:
            names_lower = [n.lower() for n in device_names]
            return [d for d in self.devices if d["deviceName"].lower() in names_lower]
        return self.devices

    def toggle_lights(self, args: ToggleLights):
        """Changes the state of all lights in a room or specific devices"""
        action = args.action.lower()

        if action not in ["on", "off"]:
            return {"status": "Failed", "reason": "Action must be either 'on' or 'off'"}

        targets = self._filter_devices(args.room, args.device_names)
        updated = []
        for light in targets:
            govee_lib.set_power(device=light["device"], model=light["model"], on=action == "on")
            updated.append(light["deviceName"])

        return {"status": "Success", "updated": updated}

    def set_brightness(self, args: SetBrightness):
        """Changes the brightness of all lights in a room or specific devices"""
        if not 0 <= args.brightness <= 100:
            return {"status": "Failed", "reason": "Brightness must be between 0 and 100"}

        targets = self._filter_devices(args.room, args.device_names)
        updated = []
        for light in targets:
            govee_lib.set_brightness(device=light["device"], model=light["model"], brightness=args.brightness)
            updated.append(light["deviceName"])

        return {"status": "Success", "updated": updated}

    def set_color(self, args: SetColor):
        """Changes the color of all lights in a room or specific devices"""
        try:
            r, g, b = govee_lib.parse_color_name(args.color)
        except ValueError as e:
            return {"status": "Failed", "reason": str(e)}

        targets = self._filter_devices(args.room, args.device_names)
        updated = []
        for light in targets:
            govee_lib.set_color(device=light["device"], model=light["model"], r=r, g=g, b=b)
            updated.append(light["deviceName"])

        return {"status": "Success", "updated": updated}


try:
    govee_controller = Govee()
except Exception:
    govee_controller = None
