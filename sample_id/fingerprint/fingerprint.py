import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def from_file(audio_path, id, sr, hop_length=512, feature="sift"):
    if feature == "sift":
        from sample_id.fingerprint import sift

        return sift.from_file(audio_path, id, sr, hop_length)
    else:
        raise NotImplementedError


def load(filepath: str):
    with np.load(filepath) as data:
        return Fingerprint(
            data["keypoints"], data["descriptors"], data["id"].item(), data["sr"].item(), data["hop"].item()
        )


class Fingerprint:
    keypoints = NotImplemented
    descriptors = NotImplemented
    spectrogram = NotImplemented
    id = NotImplemented
    sr = NotImplemented
    hop_length = NotImplemented

    def __init__(self, keypoints, descriptors, id, sr, hop_length):
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.id = id
        self.sr = sr
        self.hop_length = hop_length

    def remove_similar_keypoints(self):
        if len(self.descriptors) > 0:
            logger.info("Removing duplicate/similar keypoints...")
            a = np.array(self.descriptors)
            rounding_factor = 10
            b = np.ascontiguousarray((a // rounding_factor) * rounding_factor).view(
                np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
            )
            _, idx = np.unique(b, return_index=True)
            desc = a[sorted(idx)]
            kp = np.array([k for i, k in enumerate(self.keypoints) if i in idx])
            logger.info("Removed {} duplicate keypoints".format(a.shape[0] - idx.shape[0]))
            self.keypoints = kp
            self.descriptors = desc

    def keypoint_ms(self, kp):
        return kp[0] * self.hop_length * 1000.0 / self.sr

    def keypoint_index_ids(self):
        return np.repeat(self.id, self.keypoints.shape[0])

    def keypoint_index_ms(self):
        return np.array([self.keypoint_ms(kp) for kp in self.keypoints])

    def save_to_dir(self, dir: str, compress=True):
        filename = f"{self.id}.npz"
        filepath = os.path.join(dir, filename)
        self.save(filepath)

    def save(self, filepath: str, compress=True):
        if compress:
            save_fn = np.savez_compressed
        else:
            save_fn = np.savez
        save_fn(
            filepath,
            keypoints=self.keypoints,
            descriptors=self.descriptors,
            sr=self.sr,
            hop=self.hop_length,
            id=self.id,
        )


def save_fingerprints(fingerprints, filepath: str, compress=True):
    keypoints = np.vstack(fp.keypoints for fp in fingerprints)
    descriptors = np.vstack(fp.descriptors for fp in fingerprints)
    ids = np.hstack(np.repeat(fp.id, fp.keypoints.shape[0]) for fp in fingerprints)
    if compress:
        save_fn = np.savez_compressed
    else:
        save_fn = np.savez
    save_fn(
        filepath,
        keypoints=keypoints,
        descriptors=descriptors,
        ids=ids,
        sr=fingerprints[0].sr,
        hop=fingerprints[0].hop_length,
    )


def load_fingerprints(filepath: str):
    with np.load(filepath) as data:
        return Fingerprints(data["keypoints"], data["descriptors"], data["ids"], data["sr"].item(), data["hop"].item())


class Fingerprints:
    def __init__(self, keypoints, descriptors, ids, sr, hop_length):
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.ids = ids
        self.sr = sr
        self.hop_length = hop_length
