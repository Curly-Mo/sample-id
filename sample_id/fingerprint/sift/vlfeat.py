import logging

import cyvlfeat.sift
import numpy as np

from . import Keypoint, SiftFingerprint

logger = logging.getLogger(__name__)


class SiftVlfeat(SiftFingerprint):
    def __init__(self, audio_path, sr, id, settings, implementation="vlfeat"):
        super().__init__(audio_path, sr, id, settings)

    def sift_spectrogram(self, s, id, height, **kwargs):
        # I = np.flipud(S)
        logger.info(f"{id}: Extracting SIFT keypoints...")
        keypoints, descriptors = self.sift(s, **kwargs)
        # keypoints, descriptors = keypoints.T, descriptors.T
        logger.info(f"{id}: {len(keypoints)} keypoints found!")
        logger.info(f"{id}: Removing edge keypoints...")
        keypoints, descriptors = self.remove_edge_keypoints(keypoints, descriptors, s, height)
        logger.info(f"{id}: {len(keypoints)} keypoints remaining")

        logger.info(f"{id}: Creating keypoint objects...")
        keypoint_objs = []
        for keypoint, descriptor in zip(keypoints, descriptors):
            # cyvlfeat puts y before x
            keypoint[0], keypoint[1] = keypoint[1], keypoint[0]
            keypoint_objs.append(Keypoint(*keypoint, source=id))

        return keypoint_objs, descriptors

    def sift(
        self, S, contrast_thresh=0.1, edge_thresh=100, levels=3, magnif=3, window_size=2, first_octave=0, **kwargs
    ):
        # Scale to 0-255
        I = 255 - (S - S.min()) / (S.max() - S.min()) * 255
        keypoints, descriptors = cyvlfeat.sift.sift(
            I.astype(np.float32),
            peak_thresh=contrast_thresh,
            edge_thresh=edge_thresh,
            magnification=magnif,
            window_size=window_size,
            first_octave=first_octave,
            compute_descriptor=True,
        )
        # Add each keypoint orientation back to descriptors
        # This effectively removes rotation invariance
        # TODO: Not sure yet if this is a good idea
        # descriptors = (descriptors + [kp[3] for kp in keypoints.T])
        return keypoints, descriptors
