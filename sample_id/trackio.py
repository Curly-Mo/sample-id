import logging
import os
from typing import Iterable

import joblib
import mutagen

logger = logging.getLogger(__name__)


class Track(object):
    def __init__(self, artist, title, path=None):
        self.artist = artist
        self.title = title
        self.path = path
        self.youtube = None
        self.genre = None
        self.year = None
        self.samples = []
        self.sampled = []

    def __repr__(self):
        return "{} - {}".format(self.artist, self.title).encode("ascii", "ignore").decode()


class Sample(object):
    def __init__(self, original, derivative, type, instrument):
        self.original = original
        self.derivative = derivative
        self.type = type
        self.instrument = instrument
        self.original_times = []
        self.derivative_times = []

    def __repr__(self):
        return "{} :: {}".format(self.original, self.derivative)


def flatten_paths(paths: Iterable[str]):
    for path in paths:
        if os.path.isdir(path):
            file_paths = file_paths_in_dir(path)
            for file_path in file_paths:
                yield file_path
        else:
            yield path


def file_paths_in_dir(directory):
    for f in os.listdir(directory):
        path = os.path.join(directory, f)
        if os.path.isfile(path):
            yield path


def track_from_path(path):
    try:
        track = track_from_tags(path)
        if track.artist and track.title:
            return track
    except:
        logger.debug("Can't build from tags")
    try:
        track = track_from_filename(path)
        if track.artist and track.title:
            return track
    except:
        logger.debug("Can't build from filename")
    name = os.path.basename(path)
    name = os.path.splitext(name)[0]
    track = Track(name, name, path)
    return track


def track_from_tags(path):
    mutagen_track = mutagen.File(path)
    track = Track(
        artist=mutagen_track["artist"][0],
        title=mutagen_track["title"][0],
        path=path,
    )
    return track


def track_from_filename(path):
    basename = os.path.basename(path)
    name = os.path.splitext(basename)[0]
    parts = name.split(" - ")
    artist = parts[2].strip()
    title = parts[3].strip()
    track = Track(artist, title, path)
    return track


def parse_track_files(track_paths: Iterable[str]) -> Iterable[Track]:
    """
    track_paths could be one or more files or directories

    return an iterable of Track objects
    """
    for file_path in flatten_paths(track_paths):
        yield track_from_path(file_path)
