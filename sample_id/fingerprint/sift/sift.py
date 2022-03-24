import logging
from typing import Optional, Tuple

import librosa
import numpy as np

from sample_id import audio, fingerprint

logger = logging.getLogger(__name__)


def from_file(audio_path, id, sr, hop_length=512, implementation="vlfeat", **kwargs):
    if implementation == "vlfeat":
        from sample_id.fingerprint.sift import vlfeat

        return vlfeat.SiftVlfeat(audio_path, id, sr, hop_length, **kwargs)
    else:
        raise NotImplementedError(implementation)


class SiftFingerprint(fingerprint.Fingerprint):
    def __init__(self, audio_path, id, sr, hop_length, octave_bins=36, **kwargs):
        self.id = id
        kp, desc, s = self.sift_file(audio_path, sr, hop_length, octave_bins=octave_bins, **kwargs)
        super().__init__(kp, desc, id, sr, hop_length, octave_bins=octave_bins)
        # self.spectrogram = s

    def sift_spectrogram(self, s, id, **kwargs):
        raise NotImplementedError

    def sift_file(self, audio_path, sr, hop_length, octave_bins=36, n_octaves=6, fmin=41.2, **kwargs):
        logger.info("{}: Loading signal into memory...".format(audio_path.encode("ascii", "ignore")))
        y, sr = librosa.load(audio_path, sr=sr)
        # logger.info('{}: Trimming silence...'.format(audio_path))
        # y = np.concatenate([[0], np.trim_zeros(y), [0]])
        logger.info(f"{self.id}: Generating Spectrogram...")
        specgram = audio.cqtgram(y, sr, hop_length=hop_length, octave_bins=octave_bins, n_octaves=n_octaves, fmin=fmin)
        # s = audio.chromagram(y, hop_length=256, n_fft=4096, n_chroma=36)
        keypoints, descriptors = self.sift_spectrogram(specgram, id=self.id, **kwargs)
        keypoints, descriptors = self.remove_edge_keypoints(keypoints, descriptors, specgram, octave_bins * n_octaves)
        return keypoints, descriptors, specgram

    def remove_edge_keypoints(self, keypoints, descriptors, specgram, height):
        logger.info(f"{self.id}: Removing edge keypoints...")
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
        start = start + height / 16
        end = end - height / 4
        out_kp = []
        out_desc = []
        for keypoint, descriptor in zip(keypoints, descriptors):
            # Skip keypoints on the left and right edges of spectrogram
            if start < keypoint[0] < end:
                out_kp.append(keypoint)
                out_desc.append(descriptor)
        logger.info(f"{self.id}: Edge keypoints removed: {len(keypoints) - len(out_kp)}, remaining: {len(out_kp)}")
        return np.array(out_kp), np.array(out_desc)
