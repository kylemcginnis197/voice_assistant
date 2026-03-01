import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.exceptions import SpotifyException
from pydantic import BaseModel, Field
from typing import Optional, List

VALID_MEDIA_TYPES = {"album", "artist", "playlist", "track", "show", "episode", "audiobook"}

class PausePlayback(BaseModel):
    pass

class StartPlayback(BaseModel):
    spotify_uris: Optional[List[str]] = Field(
        description="Play a specific song, artist, album, etc by providing a URI or list of URIs. URIs can be found via the spotify search tool. Providing multiple will act as a temporary playlist. If none are provided then it will resume the current song."
    )

class NextTrack(BaseModel):
    pass

class PreviousTrack(BaseModel):
    pass

class SearchSpotify(BaseModel):
    search_query: Optional[str] = Field(
        description="The search query. Can be a song title, artist name, album name, podcast name, or any general search term. Leave empty string if using only field filters like artist/album/genre."
    )
    media_types: Optional[List[str]] = Field(
        default=["track"],
        description="List of media types to search. Options: 'track', 'album', 'artist', 'playlist', 'show' (podcast), 'episode' (podcast episode), 'audiobook'. Defaults to ['track']. Pass multiple types to search across categories at once."
    )
    artist: Optional[str] = Field(
        default=None,
        description="Narrow results to a specific artist. Appended as 'artist:<name>' in the Spotify query. Use when the user specifies an artist by name."
    )
    album: Optional[str] = Field(
        default=None,
        description="Narrow results to a specific album. Appended as 'album:<name>' in the Spotify query. Use when the user wants a track from a particular album."
    )
    genre: Optional[str] = Field(
        default=None,
        description="Narrow results to a specific genre (e.g. 'jazz', 'dream pop', 'hip-hop'). Appended as 'genre:<name>' in the Spotify query."
    )
    limit: int = Field(
        default=5,
        description="Number of results to return per media type. Range 1–10. Use a higher number when the user wants to browse options; use 1 when they want to immediately play the top result."
    )

from os import getenv

class Spotify:
    def __init__(self):
        scope = "user-modify-playback-state user-read-playback-state user-read-currently-playing playlist-read-private user-library-read user-read-recently-played"
        self.sp = spotipy.Spotify(
            auth_manager=SpotifyOAuth(
                client_id=getenv("SPOTIPY_CLIENT_ID"),
                client_secret=getenv("SPOTIPY_CLIENT_SECRET"),
                redirect_uri=getenv("SPOTIPY_REDIRECT_URI"),
                open_browser=False,
                scope=scope
            )
        )

    def pause_playback(self, args: PausePlayback):
        """Pause the user's active Spotify playback."""
        try:
            self.sp.pause_playback()
        except SpotifyException as e:
            if e.http_status == 403:
                return "Playback already paused."
            return f"Failed to pause music, reason: {e}"

    def start_playback(self, args: StartPlayback):
        """Resume the user's active Spotify playback or play songs, artists, albums, etc using a list spotify URIs obtained from search function"""
        try:
            self.sp.start_playback(uris=args.spotify_uris)
        except SpotifyException as e:
            if e.http_status == 403:
                return "Playback already playing."
            return f"Failed to start music, reason: {e}"

    def next_track(self, args: NextTrack):
        """Skip to the next track in the user's Spotify queue."""
        try:
            self.sp.next_track()
        except SpotifyException as e:
            return f"Failed to skip track, reason: {e}"

    def previous_track(self, args: PreviousTrack):
        """Go back to the previous track in the user's Spotify queue."""
        try:
            self.sp.previous_track()
        except SpotifyException as e:
            return f"Failed to go to previous track, reason: {e}"

    def search(self, args: SearchSpotify) -> dict:
        """Search Spotify for tracks, albums, artists, playlists, shows, episodes, or audiobooks.
        Returns a dict of results grouped by media type, each with a URI for playback.
        Always call this before playing something the user asked for by name.
        """
        q = args.search_query if args.search_query is not None else ""
        if args.artist: q += f" artist:{args.artist}"
        if args.album:  q += f" album:{args.album}"
        if args.genre:  q += f" genre:{args.genre}"

        types = list({t.lower() for t in args.media_types if t.lower() in VALID_MEDIA_TYPES}) or ["track"]
        limit = max(1, min(10, args.limit))

        try:
            results = self.sp.search(q=q, limit=limit, type=",".join(types))
        except SpotifyException as e:
            return {"status": "Failed", "reason": str(e)}

        out = {}

        if "tracks" in results:
            out["tracks"] = [
                {
                    "name": t["name"],
                    "artists": [a["name"] for a in t["artists"]],
                    "album": t["album"]["name"],
                    "uri": t["uri"],
                    "duration_ms": t["duration_ms"],
                }
                for t in results["tracks"]["items"]
            ]

        if "albums" in results:
            out["albums"] = [
                {
                    "name": a["name"],
                    "artists": [ar["name"] for ar in a["artists"]],
                    "uri": a["uri"],
                    "total_tracks": a["total_tracks"],
                    "release_date": a["release_date"],
                }
                for a in results["albums"]["items"]
            ]

        if "artists" in results:
            out["artists"] = [
                {
                    "name": a["name"],
                    "uri": a["uri"],
                    "genres": a.get("genres", []),
                }
                for a in results["artists"]["items"]
            ]

        if "playlists" in results:
            out["playlists"] = [
                {
                    "name": p["name"],
                    "owner": p["owner"]["display_name"],
                    "uri": p["uri"],
                    "total_tracks": p["tracks"]["total"],
                }
                for p in results["playlists"]["items"] if p
            ]

        if "shows" in results:
            out["shows"] = [
                {
                    "name": s["name"],
                    "publisher": s["publisher"],
                    "uri": s["uri"],
                }
                for s in results["shows"]["items"]
            ]

        if "episodes" in results:
            out["episodes"] = [
                {
                    "name": e["name"],
                    "show": e.get("show", {}).get("name"),
                    "uri": e["uri"],
                    "duration_ms": e.get("duration_ms"),
                }
                for e in results["episodes"]["items"]
            ]

        if "audiobooks" in results:
            out["audiobooks"] = [
                {
                    "name": b["name"],
                    "authors": [a["name"] for a in b.get("authors", [])],
                    "uri": b["uri"],
                }
                for b in results["audiobooks"]["items"]
            ]

        return out


try:
    sp = Spotify()
except Exception:
    sp = None
