# Gradio seems to truncate files without keeping the extension, so we need to truncate the file prefix ourself 
import os
import pathlib
from typing import List
import zipfile

import ffmpeg
from more_itertools import unzip

from src.download import ExceededMaximumDuration, download_url

MAX_FILE_PREFIX_LENGTH = 17

class AudioSource:
    def __init__(self, source_path, source_name = None, audio_duration = None):
        self.source_path = source_path
        self.source_name = source_name
        self._audio_duration = audio_duration

        # Load source name if not provided
        if (self.source_name is None):
            file_path = pathlib.Path(self.source_path)
            self.source_name = file_path.name

    def get_audio_duration(self):
        if self._audio_duration is None:
            self._audio_duration = float(ffmpeg.probe(self.source_path)["format"]["duration"])

        return self._audio_duration

    def get_full_name(self):
        return self.source_name

    def get_short_name(self, max_length: int = MAX_FILE_PREFIX_LENGTH):
        file_path = pathlib.Path(self.source_name)
        short_name = file_path.stem[:max_length] + file_path.suffix

        return short_name

    def __str__(self) -> str:
        return self.source_path

class AudioSourceCollection:
    def __init__(self, sources: List[AudioSource]):
        self.sources = sources

    def __iter__(self):
        return iter(self.sources)

def resolve_gradio_file(value) -> str:
    """
    Gradio has changed what a file component yields between major versions: a tempfile wrapper with
    a .name in 3.x, a plain path string in 4.x and later, and a dict in some API/queue paths.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("path", "name", "url"):
            if value.get(key):
                return str(value[key])
        return None
    for attribute in ("path", "name"):
        if hasattr(value, attribute):
            return getattr(value, attribute)
    return str(value)

def get_audio_source_collection(urlData: str, multipleFiles: List, microphoneData: str, input_audio_max_duration: float = -1) -> List[AudioSource]:
    output: List[AudioSource] = []

    if urlData:
        # Download from YouTube. This could also be a playlist or a channel.
        output.extend([ AudioSource(x) for x in download_url(urlData, input_audio_max_duration, playlistItems=None) ])
    else:
        # Add input files
        if (multipleFiles is not None):
            if not isinstance(multipleFiles, list):
                multipleFiles = [multipleFiles]
            output.extend([ AudioSource(path) for path in map(resolve_gradio_file, multipleFiles) if path ])
        if (microphoneData is not None):
            output.append(AudioSource(resolve_gradio_file(microphoneData)))

    total_duration = 0

    # Calculate total audio length. We do this even if input_audio_max_duration
    # is disabled to ensure that all the audio files are valid.
    for source in output:
        audioDuration = ffmpeg.probe(source.source_path)["format"]["duration"]
        total_duration += float(audioDuration)
        
        # Save audio duration
        source._audio_duration = float(audioDuration)

    # Ensure the total duration of the audio is not too long
    if input_audio_max_duration > 0:
        if float(total_duration) > input_audio_max_duration:
            raise ExceededMaximumDuration(videoDuration=total_duration, maxDuration=input_audio_max_duration, message="Video(s) is too long")

    # Return a list of audio sources
    return output