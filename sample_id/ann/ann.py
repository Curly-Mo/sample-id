from __future__ import annotations

import abc
import logging
import os
import tempfile
import zipfile
from typing import Iterable, Optional

import numpy as np

from sample_id.fingerprint import Fingerprint

logger = logging.getLogger(__name__)


MATCHER_FILENAME: str = "matcher.ann"
META_FILENAME: str = "meta.npz"


# TODO: Make this a proper interface, for now just implementing annoy
class Matcher(abc.ABC):
    """Nearest neighbor matcher that may use one of various implementations under the hood."""

    def __init__(self, metadata: MatcherMetadata):
        self.index = 0
        self.meta = metadata
        self.model = self.init_model()

    @abc.abstractmethod
    def init_model(self) -> Any:
        """Initialize the model."""
        pass

    @abc.abstractmethod
    def save_model(self, filepath: str) -> str:
        """Save this matcher's model to disk."""
        pass

    @abc.abstractmethod
    def load_model(self, filepath: str) -> Any:
        """Load this matcher's model from disk."""
        pass

    def add_fingerprint(self, fingerprint: Fingerprint, dedupe=True) -> Matcher:
        """Add a Fingerprint to the matcher."""
        if self.can_add_fingerprint(fingerprint):
            if dedupe:
                fingerprint.remove_similar_keypoints()
            logger.info(f"Adding {fingerprint} to index.")
            self.meta.index_to_id = np.hstack([self.meta.index_to_id, fingerprint.keypoint_index_ids()])
            self.meta.index_to_ms = np.hstack([self.meta.index_to_ms, fingerprint.keypoint_index_ms()])
            self.meta.index_to_kp = np.vstack([self.meta.index_to_kp, fingerprint.keypoints])
            for descriptor in fingerprint.descriptors:
                self.model.add_item(self.index, descriptor)
                self.index += 1
        return self

    def add_fingerprints(self, fingerprints: Iterable[Fingerprint], **kwargs) -> Matcher:
        """Add Fingerprints to the matcher."""
        for fingerprint in fingerprints:
            self.add_fingerprint(fingerprint, **kwargs)
        return self

    def can_add_fingerprint(self, fingerprint: Fingerprint) -> Boolean:
        """Check if fingerprint can be added to matcher."""
        if not self.meta.sr:
            self.meta.sr = fingerprint.sr
        if not self.meta.hop_length:
            self.meta.hop_length = fingerprint.hop_length
        if self.meta.sr != fingerprint.sr:
            logger.warn(f"Can't add fingerprint with sr={fingerprint.sr}, must equal matcher sr={self.meta.sr}")
        if self.meta.hop_length != fingerprint.hop_length:
            logger.warn(
                f"Can't add fingerprint with hop_length={fingerprint.hop_length}, must equal matcher hop_length={self.meta.hop_length}"
            )
        return True

    def save(self, filepath: str, compress: bool = True) -> None:
        """Save this matcher to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_model_path = os.path.join(tmpdir, MATCHER_FILENAME)
            tmp_meta_path = os.path.join(tmpdir, META_FILENAME)
            tmp_model_path = self.save_model(tmp_model_path)
            self.meta.save(tmp_meta_path)
            with zipfile.ZipFile(filepath, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
                logger.info(f"Zipping {tmp_model_path} and {tmp_meta_path} into {zipf.filename}")
                zipf.write(tmp_model_path, arcname=MATCHER_FILENAME)
                zipf.write(tmp_meta_path, arcname=META_FILENAME)

    def unload(self) -> None:
        """Unload things from memory and cleanup any temporary files."""
        self.model.unload()
        if "tempdir" in vars(self):
            self.tempdir.cleanup()

    @classmethod
    def create(cls, sr: Optional[int] = None, hop_length: Optional[int] = None, **kwargs) -> Matcher:
        """Create an instance, pass any kwargs needed by the subclass."""
        meta = MatcherMetadata(sr=sr, hop_length=hop_length, **kwargs)
        return cls(meta)

    @classmethod
    def from_fingerprint(cls, fp: Fingerprint, **kwargs) -> Matcher:
        """Useful for determining metadata for the Matcher based on the data being added."""
        matcher = cls.create(fp.descriptors.shape[1], sr=fp.sr, hop_length=fp.hop_length, **kwargs)
        return matcher.add_fingerprint(fp, **kwargs)

    @classmethod
    def from_fingerprints(cls, fingerprints: Iterable[Fingerprint], **kwargs) -> Matcher:
        """My data is small, just create and train the entire matcher."""
        fp = fingerprints[0]
        matcher = cls.create(fp.descriptors.shape[1], sr=fp.sr, hop_length=fp.hop_length, **kwargs)
        return matcher.add_fingerprints(fingerprints, **kwargs)

    @classmethod
    def load(cls, filepath: str) -> Matcher:
        """Load a matcher from disk."""
        with zipfile.ZipFile(filepath, "r") as zipf:
            tempdir = tempfile.TemporaryDirectory()
            tmp_model_path = os.path.join(tempdir.name, MATCHER_FILENAME)
            tmp_meta_path = os.path.join(tempdir.name, META_FILENAME)
            logger.info(f"Extracting matcher model to {tmp_model_path}.")
            zipf.extract(MATCHER_FILENAME, tempdir.name)
            logger.info(f"Extracting matcher metadata to {tmp_meta_path}.")
            zipf.extract(META_FILENAME, tempdir.name)
            meta = MatcherMetadata.load(tmp_meta_path)
            matcher = cls(meta)
            matcher.tempdir = tempdir
            matcher.load_model(tmp_model_path)
            return matcher


class MatcherMetadata:
    """Metadata for a Matcher object."""

    def __init__(
        self,
        sr: Optional[int] = None,
        hop_length: Optional[int] = None,
        index_to_id=None,
        index_to_ms=None,
        index_to_kp=None,
        **kwargs,
    ):
        self.sr = sr
        self.hop_length = hop_length
        self.index_to_id = index_to_id
        self.index_to_ms = index_to_ms
        self.index_to_kp = index_to_kp
        if index_to_id is None:
            self.index_to_id = np.array([], str)
        if index_to_ms is None:
            self.index_to_ms = np.array([], np.uint32)
        if index_to_kp is None:
            self.index_to_kp = np.empty(shape=(0, 4), dtype=np.float32)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def save(self, filepath: str, compress: bool = True) -> None:
        """Save this matcher's metadata to disk."""
        save_fn = np.savez_compressed if compress else np.savez
        logger.info(f"Saving matcher metadata to {filepath}...")
        save_fn(
            filepath,
            n_features=self.n_features,
            metric=self.metric,
            sr=self.sr,
            hop_length=self.hop_length,
            index_to_id=self.index_to_id,
            index_to_ms=self.index_to_ms,
            index_to_kp=self.index_to_kp,
        )

    @classmethod
    def load(cls, filepath: str) -> MatcherMetadata:
        """Load this matcher's metadata from disk."""
        logger.info(f"Loading matcher metadata from {filepath}...")
        with np.load(filepath) as data:
            meta = cls(
                n_features=data["n_features"].item(),
                metric=data["metric"].item(),
                sr=data["sr"].item(),
                hop_length=data["hop_length"].item(),
                index_to_id=data["index_to_id"],
                index_to_ms=data["index_to_ms"],
                index_to_kp=data["index_to_kp"],
            )
            logger.info(f"Loaded metadata: {meta}")
            return meta

    def __repr__(self):
        attrs = ",".join(f"{k}={v}" for k, v in vars(self).items() if type(v) in (int, float, bool, str))
        return f"MatcherMeta({attrs})"
