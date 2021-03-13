import logging
from typing import Optional, Tuple

import librosa
import numpy as np

from sample_id import audio, fingerprint

logger = logging.getLogger(__name__)


class Keypoint(object):
    def __init__(self, x, y, scale=None, orientation=None, source=None, descriptor=None):
        self.x = x
        self.y = y
        self.scale = scale
        self.orientation = orientation
        self.source = source
        self.descriptor = descriptor

    @property
    def kp(self) -> Tuple[float, float, Optional[float], Optional[float]]:
        return np.array([self.x, self.y, self.scale, self.orientation])


def from_file(audio_path, sr, id, settings, implementation="vlfeat"):
    if implementation == "vlfeat":
        from sample_id.fingerprint.sift import vlfeat

        return vlfeat.SiftVlfeat(audio_path, sr, id, settings)
    else:
        raise NotImplementedError(implementation)


class SiftFingerprint(fingerprint.Fingerprint):
    def __init__(self, audio_path, sr, id, settings, implementation="vlfeat"):
        self.audio_path = audio_path
        self.sr = sr
        self.id = id.encode("ascii", "ignore")
        self.settings = settings
        kp, desc, s = self.sift_file(**settings)
        self.keypoints = kp
        self.descriptors = desc
        self.spectrogram = s

    def sift_spectrogram(self, s, id, height, **kwargs):
        raise NotImplementedError

    def sift_file(self, hop_length=512, octave_bins=24, n_octaves=8, fmin=40, **kwargs):
        logger.info("{}: Loading signal into memory...".format(self.audio_path.encode("ascii", "ignore")))
        y, sr = librosa.load(self.audio_path, sr=self.sr)
        # logger.info('{}: Trimming silence...'.format(audio_path))
        # y = np.concatenate([[0], np.trim_zeros(y), [0]])
        logger.info("{}: Generating Spectrogram...".format(self.id))
        specgram = audio.cqtgram(y, sr, hop_length=hop_length, octave_bins=octave_bins, n_octaves=n_octaves, fmin=fmin)
        # s = audio.chromagram(y, hop_length=256, n_fft=4096, n_chroma=36)
        keypoints, descriptors = self.sift_spectrogram(specgram, id=self.id, height=octave_bins * n_octaves, **kwargs)
        return keypoints, descriptors, specgram

    def remove_edge_keypoints(self, keypoints, descriptors, specgram, height):
        logger.info("Removing edge keypoints...")
        min_value = np.min(specgram)
        start = next(
            (index for index, frame in enumerate(specgram.T) if sum(value > min_value for value in frame) > height / 2),
            0,
        )
        end = specgram.shape[1] - next(
            (
                index
                for index, frame in enumerate(reversed(specgram.T))
                if sum(value > min_value for value in frame) > height / 2
            ),
            0,
        )
        start = start + 10
        end = end - 10
        out_kp = []
        out_desc = []
        for keypoint, descriptor in zip(keypoints, descriptors):
            # Skip keypoints on the left and right edges of spectrogram
            if start < keypoint[1] < end:
                out_kp.append(keypoint)
                out_desc.append(descriptor)
        logger.info("Edge keypoints removed: {}, remaining: {}".format(len(keypoints) - len(out_kp), len(out_kp)))
        return out_kp, out_desc
