"""Acoustic fingerprinting for Sample Identification"""
import logging
from typing import Iterable

import numpy as np

from sample_id import ann
from sample_id.ann import annoy
from sample_id.fingerprint import fingerprint
from sample_id.io import Track

logger = logging.getLogger(__name__)


def train_keypoints(
    tracks: Iterable[Track],
    hop_length=256,
    octave_bins=36,
    n_octaves=7,
    fmin=50,
    sr=22050,
    algorithm="annoy",
    dedupe=False,
    save=None,
    save_spec=False,
    **kwargs,
):
    # TODO: fix this settings garbage
    settings = locals().copy()
    for key in settings["kwargs"]:
        settings[key] = settings["kwargs"][key]
    del settings["kwargs"]
    del settings["tracks"]
    logger.info("Settings: {}".format(settings))

    spectrograms = {}
    # keypoints = []
    # descriptors = []
    model_tracks = {}
    fingerprints = []
    for track in tracks:
        track_id = str(track)
        model_tracks[track_id] = track
        fp = fingerprint.from_file(track.path, track_id, sr, dedupe=dedupe, **settings)
        fingerprints.extend(fp)

        # if save_spec:
        #     path = save_spectrogram(fp.spectrogram, track_id, save)
        #     spectrograms[track_id] = path
        # keypoints.extend(fp.keypoints)
        # descriptors.extend(fp.descriptors)
    # descriptors = np.vstack(descriptors)
    # matcher = annoy.AnnoyMatcher.create(descriptors, algorithm=algorithm)
    matcher = annoy.AnnoyMatcher.from_fingerprints(fingerprints)
    # descriptors = None
    # model = Model(matcher, keypoints, settings, tracks=model_tracks)
    # model.spectrograms = spectrograms
    # if save:
    #     save_model(model, save)
    return matcher
