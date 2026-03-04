import spotipy
import time
from spotipy.oauth2 import SpotifyOAuth
from spotipy.exceptions import SpotifyException
from pydantic import BaseModel, Field
from typing import Optional, List

VALID_MEDIA_TYPES = {"album", "artist", "track"}

class BaseSpotify(BaseModel):
    pass

class StartPlayback(BaseModel):
    spotify_uris: Optional[List[str]] = Field(
        description="Play a specific song, artist, album, etc by providing a URI or list of URIs. URIs can be found via the spotify search tool. Providing multiple will act as a temporary playlist. If none are provided then it will resume the current song."
    )

class SearchSpotify(BaseModel):
    search_query: Optional[str] = Field(
        description="The search query. Can be a song title, artist name, album name, podcast name, or any general search term. Leave empty string if using only field filters like artist/album/genre."
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
        description="Number of results to return per media type. Range 1–10. Use a higher limit to have more options to pick from."
    )

from os import getenv

class Spotify:
    def __init__(self):
        scope = "user-modify-playback-state user-read-playback-state user-read-currently-playing playlist-read-private user-library-read user-read-recently-played user-follow-read"
        self.sp = spotipy.Spotify(
            auth_manager=SpotifyOAuth(
                client_id=getenv("SPOTIPY_CLIENT_ID"),
                client_secret=getenv("SPOTIPY_CLIENT_SECRET"),
                redirect_uri=getenv("SPOTIPY_REDIRECT_URI"),
                open_browser=False,
                scope=scope
            )
        )

    def current_playback(self, args: BaseSpotify = None):
        """
        Works with the Spotify Song/Music API.

        Args:
            BaseSpotify

        Returns details on whether a song is currently playing and the current song (or last song if nothing is actively playing).
        """

        try:
            data = self.sp.current_playback()
        except SpotifyException as e:
            return f"Failed to get current playback state, reason: {e}"
        
        if data is None:
            return "Failed to get current playback state, data is none from spotify api."

        device = data.get("device", {})
        item = data.get("item", {})

        return {
            "device": device.get("name", "N/A"),
            "is_playing": data.get("is_playing", False),
            "media_name": item.get("name", None),
            "media_type": item.get("type", None),
            "media_uri": item.get("uri", None),
        }

    def pause_playback(self, args: BaseSpotify = None):
        """
        Works with the Spotify Song/Music API.

        Args:
            BaseSpotify

        Pause the user's active Spotify playback.
        """
        try:
            self.sp.pause_playback()
        except SpotifyException as e:
            if e.http_status == 403:
                return "Playback already paused."
            return f"Failed to pause music, reason: {e}"
        
        # current playback takes a little time to update.
        time.sleep(0.2)

        return {"state": "looks like action was carried out", "current_playback_state": self.current_playback()}


    def start_playback(self, args: StartPlayback):
        """
        Works with the Spotify Song/Music API.

        Args: 
            spotify_uris: Optional[list[str]]

        1. If no spotify_uris are provided, it will try to resume playback.
        2. If spotify song uris are provided, then it will play the song for the user.
        """
        try:
            self.sp.start_playback(uris=args.spotify_uris)
        except SpotifyException as e:
            if e.http_status == 403:
                return "Playback already playing."
            return f"Failed to start music, reason: {e}"
        
        # current playback takes a little time to update.
        time.sleep(0.2)

        return {"state": "looks like action was carried out", "current_playback_state": self.current_playback()}

    def next_track(self, args: BaseSpotify = None):
        """
        Works with the Spotify Song/Music API.

        Args:
            BaseSpotify

        Skip to the next track in the user's Spotify queue. 
        """
        try:
            self.sp.next_track()
        except SpotifyException as e:
            return f"Failed to skip track, reason: {e}"

        # current playback takes a little time to update.
        time.sleep(0.2)
        
        return {"state": "looks like action was carried out", "current_playback_state": self.current_playback()}

    def previous_track(self, args: BaseSpotify = None):
        """
        Works with the Spotify Song/Music API.

        Args:
            BaseSpotify

        When the user is listening to music, it plays the last song the user listened to.
        """
        try:
            self.sp.previous_track()
        except SpotifyException as e:
            return f"Failed to go to previous track, reason: {e}"
        
        # current playback takes a little time to update.
        time.sleep(0.2)

        return {"state": "looks like action was carried out", "current_playback_state": self.current_playback()}

    def search(self, args: SearchSpotify) -> dict:
        """
        Works with the Spotify Song/Music API.
        
        Args:
            SearchSpotify class
        
        Search Spotify for tracks can be filtered by artist, album, and genre.
        Returns a dict of results grouped by media type, each with a URI for playback.
        Always call this before playing something the user asked for by name.

        Tip: Use a higher limit instead of making multiple tool calls.
        """
        q = args.search_query if args.search_query is not None else ""
        if args.artist: q += f" artist:{args.artist}"
        if args.album:  q += f" album:{args.album}"
        if args.genre:  q += f" genre:{args.genre}"

        limit = max(1, min(10, args.limit))

        try:
            results = self.sp.search(q=q, limit=limit, type="track")
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


        return out

    def get_recently_played_songs(self, args: BaseSpotify = None):
        """
        Uses the Spotify Song/Music API.

        Args:
            BaseSpotify

        Gets the previous 50 songs played on the user's spotify account. Additionally, track URI's can be used to play the song using start_playback tool.
        """
        try:
            results = self.sp.current_user_recently_played(limit=50)
        except SpotifyException as e:
            return f"Failed to get recently played songs, reason: {e}"
        
        if items := results.get("items", None):
            user_tracks = []

            for playback in items:
                if track := playback.get("track", None):
                    song_name = track.get("name", None)
                    song_uri = track.get("uri", None)
                    
                    if song_name is None or song_uri is None:
                        continue

                    album = track.get("album", {})
                    artists = [
                        {
                            "name": artist.get("name"),
                            "uri": artist.get("uri"),
                        } for artist in track.get("artists", [])
                    ]
                
                    user_tracks.append({
                        "song_name": track.get("name", "n/a"),
                        "song_uri": track.get("uri", "n/a"),
                        "artists": artists,
                        "album_name": album.get("name", "n/a"),
                        "album_uri": album.get("uri", "n/a")
                    })

            return user_tracks

sp = Spotify()
