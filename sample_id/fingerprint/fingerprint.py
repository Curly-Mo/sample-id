from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from sample_id import util

logger = logging.getLogger(__name__)


def from_file(audio_path, id, sr, hop_length=512, feature="sift", dedupe=False, **kwargs) -> Fingerprint:
    """Generate a fingerprint from an audio file."""
    if feature == "sift":
        from . import sift

        fp = sift.from_file(audio_path, id, sr, hop_length=hop_length, **kwargs)
        if dedupe:
            fp.remove_similar_keypoints()
        return fp
    else:
        raise NotImplementedError


def load(filepath: str) -> Fingerprint:
    """Load a fingerprint from file."""
    with np.load(filepath) as data:
        constructor_args = Fingerprint.__init__.__code__.co_varnames[1:]
        # TODO: replace this hack to be backwards compatible with misnamed hop arg
        if "hop" in data.keys():
            constructor_args = ["hop" if arg == "hop_length" else arg for arg in constructor_args]
        arg_data = tuple(data.get(arg) for arg in constructor_args)
        arg_values = [arg if arg is None or arg.shape else arg.item() for arg in arg_data]
        return Fingerprint(*arg_values)


class Fingerprint:
    spectrogram = NotImplemented

    def __init__(self, keypoints, descriptors, id, sr, hop_length, is_deduped=False, octave_bins=None):
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.id = id
        self.sr = sr
        self.hop_length = hop_length
        self.is_deduped = is_deduped
        self.size = len(keypoints)
        self.octave_bins = octave_bins

    def remove_similar_keypoints(self):
        if len(self.descriptors) > 0:
            logger.info(f"{self.id}: Removing duplicate/similar keypoints...")
            a = np.array(self.descriptors)
            rounding_factor = 10
            b = np.ascontiguousarray((a // rounding_factor) * rounding_factor).view(
                np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
            )
            _, idx = np.unique(b, return_index=True)
            desc = a[sorted(idx)]
            kp = np.array([k for i, k in enumerate(self.keypoints) if i in idx])
            logger.info(f"{self.id}: Removed {a.shape[0] - idx.shape[0]} duplicate keypoints")
            self.keypoints = kp
            self.descriptors = desc
            self.is_deduped = True

    def keypoint_ms(self, kp) -> int:
        return int(kp[0] * self.hop_length * 1000.0 / self.sr)

    def keypoint_index_ids(self):
        return np.repeat(self.id, self.keypoints.shape[0])

    def keypoint_index_ms(self):
        return np.array([self.keypoint_ms(kp) for kp in self.keypoints], dtype=np.uint32)

    def save_to_dir(self, dir: str, compress: bool = True):
        filepath = os.path.join(dir, self.id)
        self.save(filepath, compress=compress)

    def save(self, filepath: str, compress: bool = True):
        save_fn = np.savez_compressed if compress else np.savez
        # save all attributes used in constructor
        constructor_arg_names = self.__init__.__code__.co_varnames[1:]
        constructor_kwargs = {name: getattr(self, name, None) for name in constructor_arg_names}
        constructor_kwargs = {key: value for key, value in constructor_kwargs.items() if value is not None}
        save_fn(filepath, **constructor_kwargs)

    def __repr__(self):
        return util.class_repr(self)


def save_fingerprints(fingerprints: Iterable[Fingerprint], filepath: str, compress=True):
    # TODO: try structured arrays: https://docs.scipy.org/doc/numpy-1.13.0/user/basics.rec.html
    keypoints = np.vstack([fp.keypoints for fp in fingerprints])
    descriptors = np.vstack([fp.descriptors for fp in fingerprints])
    index_to_id = np.hstack([fp.keypoint_index_ids() for fp in fingerprints])
    # index_to_ms = np.hstack([fp.keypoint_index_ms() for fp in fingerprints])
    sr = next(fp.sr for fp in fingerprints)
    hop_length = next(fp.hop_length for fp in fingerprints)
    save_fn = np.savez_compressed if compress else np.savez
    save_fn(
        filepath,
        keypoints=keypoints,
        descriptors=descriptors,
        index_to_id=index_to_id,
        # index_to_ms=index_to_ms,
        sr=sr,
        hop=hop_length,
    )


def load_fingerprints(filepath: str) -> Fingerprints:
    with np.load(filepath) as data:
        return Fingerprints(data["keypoints"], data["descriptors"], data["index_to_id"], data["index_to_ms"])


class Fingerprints:
    def __init__(self, keypoints, descriptors, index_to_id, index_to_ms):
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.index_to_id = index_to_id
        self.index_to_ms = index_to_ms


class LazyFingerprints(Fingerprints):
    def __init__(self, npz_filepath: str):
        self.data = np.load(npz_filepath, mmap_mode="r")

    @property
    def keypoints(self):
        return self.data["keypoints"]

    @property
    def descriptors(self):
        return self.data["descriptors"]

    @property
    def index_to_id(self):
        return self.data["index_to_id"]

    @property
    def index_to_ms(self):
        return self.data["index_to_ms"]


@dataclass(unsafe_hash=True)
class Keypoint:
    """A fingerprint keypoint."""

    kp: np.ndarray[np.float32] = field(repr=False, compare=False)
    x: float = field(init=False)
    y: float = field(init=False)
    scale: float = field(init=False)
    orientation: float = field(init=False)

    def __post_init__(self):
        self.x = self.kp[0].item()
        self.y = self.kp[1].item()
        self.scale = self.kp[2].item()
        self.orientation = self.kp[3].item()
