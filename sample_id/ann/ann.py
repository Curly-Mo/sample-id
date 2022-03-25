from __future__ import annotations

import abc
import bisect
import dataclasses
import datetime
import itertools
import logging
import math
import os
import pathlib
import statistics
import tempfile
from collections import defaultdict
from dataclasses import InitVar, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from sample_id import util
from sample_id.fingerprint import Fingerprint

from . import query

logger = logging.getLogger(__name__)


MODEL_FILENAME: str = "matcher.ann"
META_FILENAME: str = "meta.npz"


# TODO: Make this a proper interface, for now just implementing annoy
class Matcher(abc.ABC):
    """Nearest neighbor matcher that may use one of various implementations under the hood."""

    def __init__(self, metadata: MatcherMetadata):
        self.index = 0
        self.num_items = 0
        self.meta = metadata
        self.model = self.init_model()

    @abc.abstractmethod
    def init_model(self) -> Any:
        """Initialize the model."""
        pass

    @abc.abstractmethod
    def save_model(self, filepath: str, **kwargs) -> str:
        """Save this matcher's model to disk."""
        pass

    @abc.abstractmethod
    def load_model(self, filepath: str, **kwargs) -> Any:
        """Load this matcher's model from disk."""
        pass

    @abc.abstractmethod
    def nearest_neighbors(self, fp: Fingerprint, k: int = 1) -> Iterable[query.Match]:
        """Fetch nearest neighbors to this fingerprint's keypoints."""
        pass

    def add_fingerprint(self, fingerprint: Fingerprint, dedupe=True) -> Matcher:
        """Add a Fingerprint to the matcher."""
        if self.can_add_fingerprint(fingerprint):
            if dedupe and not fingerprint.is_deduped:
                fingerprint.remove_similar_keypoints()
            logger.info(f"Adding {fingerprint} to index.")
            self.meta.index_to_id = np.hstack([self.meta.index_to_id, fingerprint.keypoint_index_ids()])
            # self.meta.index_to_ms = np.hstack([self.meta.index_to_ms, fingerprint.keypoint_index_ms()])
            self.meta.index_to_kp = np.vstack([self.meta.index_to_kp, fingerprint.keypoints])
            for descriptor in fingerprint.descriptors:
                self.model.add_item(self.index, descriptor)
                self.index += 1
            self.num_items += 1
        return self

    def add_fingerprints(self, fingerprints: Iterable[Fingerprint], **kwargs) -> Matcher:
        """Add Fingerprints to the matcher."""
        for fingerprint in fingerprints:
            self.add_fingerprint(fingerprint, **kwargs)
        return self

    def can_add_fingerprint(self, fingerprint: Fingerprint) -> bool:
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

    def save(
        self,
        filepath: str,
        compress: bool = True,
        tar_compression: str = "gz",
        **kwargs,
    ) -> str:
        """Save this matcher to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger.info(f"Saving {self} to temporary dir: {tmpdir}")
            tmp_model_path = os.path.join(tmpdir, MODEL_FILENAME)
            tmp_meta_path = os.path.join(tmpdir, META_FILENAME)
            tmp_model_path = self.save_model(tmp_model_path, **kwargs)
            self.meta.save(tmp_meta_path, compress=compress)
            logger.debug(f"Model file {tmp_model_path} size: {util.filesize(tmp_model_path)}")
            logger.debug(f"Metadata file {tmp_meta_path} size: {util.filesize(tmp_meta_path)}")
            logger.info(f"Zipping {[tmp_model_path, tmp_meta_path]} into {filepath}")
            util.tar_files(
                filepath,
                [tmp_model_path, tmp_meta_path],
                [MODEL_FILENAME, META_FILENAME],
                compression=tar_compression,
            )
        logger.debug(f"Tar file {filepath} size: {util.filesize(filepath)}")
        return filepath

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
        matcher = cls.create(sr=fp.sr, hop_length=fp.hop_length, n_features=fp.descriptors.shape[1], **kwargs)
        return matcher.add_fingerprint(fp, **kwargs)

    @classmethod
    def from_fingerprints(cls, fingerprints: Sequence[Fingerprint], **kwargs) -> Matcher:
        """My data is small, just create and train the entire matcher."""
        fp = fingerprints[0]
        matcher = cls.create(sr=fp.sr, hop_length=fp.hop_length, n_features=fp.descriptors.shape[1], **kwargs)
        return matcher.add_fingerprints(fingerprints, **kwargs)

    @classmethod
    def load(cls, filepath: str, dirname: Optional[str] = None, tar_compression: str = "gz", **kwargs) -> Matcher:
        """Load a matcher from disk."""
        if dirname:
            pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
            tempdir = tempfile.TemporaryDirectory(dir=dirname)
            tempdir.name = dirname
        else:
            tempdir = tempfile.TemporaryDirectory()
        logger.debug(f"Unzipping {filepath} to {tempdir.name}...")
        util.untar_members(filepath, [MODEL_FILENAME, META_FILENAME], tempdir.name, compression=tar_compression)
        return cls.load_from_dir(tempdir.name, **kwargs)

    @classmethod
    def load_from_dir(cls, dirname: str, **kwargs) -> Matcher:
        tempdir = tempfile.TemporaryDirectory(dir=dirname)
        tempdir.name = dirname
        model_path = os.path.join(tempdir.name, MODEL_FILENAME)
        meta_path = os.path.join(tempdir.name, META_FILENAME)
        meta = MatcherMetadata.load(meta_path)
        matcher = cls(meta)
        matcher.tempdir = tempdir
        matcher.load_model(model_path, **kwargs)
        return matcher

    def __repr__(self):
        return f"{self.__class__.__name__}({self.meta})"

    def filter_matches(
        self,
        matches: List[query.Match],
        abs_thresh: Optional[float] = 0.25,
        ratio_thresh: Optional[float] = None,
        cluster_dist: float = 4.0,
        cluster_size: int = 2,
        match_orientation: bool = True,
        ordered: bool = False,
        cluster_filter: Optional[Callable[[query.Cluster], bool]] = None,
    ) -> List[query.Cluster]:
        cluster_sample_dist = int(cluster_dist * self.meta.sr / self.meta.hop_length)
        return query.filter_matches(
            matches,
            abs_thresh=abs_thresh,
            ratio_thresh=ratio_thresh,
            cluster_dist=cluster_sample_dist,
            cluster_size=cluster_size,
            match_orientation=match_orientation,
            ordered=ordered,
            cluster_filter=cluster_filter,
        )

    def find_samples(
        self,
        fp: Fingerprint,
        k: int = 1,
        abs_thresh: Optional[float] = 0.25,
        ratio_thresh: Optional[float] = None,
        cluster_dist: float = 20.0,
        cluster_size: int = 2,
        match_orientation: bool = True,
        ordered: bool = False,
        cluster_filter: Optional[Callable[[query.Cluster], bool]] = None,
    ) -> query.Result:
        matches = self.nearest_neighbors(fp, k)
        clusters = self.filter_matches(
            matches,
            abs_thresh=abs_thresh,
            ratio_thresh=ratio_thresh,
            cluster_dist=cluster_dist,
            cluster_size=cluster_size,
            match_orientation=match_orientation,
            ordered=ordered,
            cluster_filter=cluster_filter,
        )
        return query.Result(fp, clusters)


class MatcherMetadata:
    """Metadata for a Matcher object."""

    def __init__(
        self,
        sr: Optional[int] = None,
        hop_length: Optional[int] = None,
        index_to_id=None,
        # index_to_ms=None,
        index_to_kp=None,
        **kwargs,
    ):
        self.sr = sr
        self.hop_length = hop_length
        self.index_to_id = index_to_id
        # self.index_to_ms = index_to_ms
        self.index_to_kp = index_to_kp
        if index_to_id is None:
            self.index_to_id = np.array([], str)
        # if index_to_ms is None:
        #     self.index_to_ms = np.array([], np.uint32)
        if index_to_kp is None:
            self.index_to_kp = np.empty(shape=(0, 4), dtype=np.float32)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def save(self, filepath: str, compress: bool = True) -> None:
        """Save this matcher's metadata to disk."""
        save_fn = np.savez_compressed if compress else np.savez
        logger.info(f"Saving metadata {self} to {filepath}...")
        save_fn(
            filepath,
            n_features=self.n_features,
            metric=self.metric,
            sr=self.sr,
            hop_length=self.hop_length,
            index_to_id=self.index_to_id,
            # index_to_ms=self.index_to_ms,
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
                # index_to_ms=data["index_to_ms"],
                index_to_kp=data["index_to_kp"],
            )
            logger.info(f"Loaded metadata: {meta}")
            return meta

    def __repr__(self) -> str:
        return util.class_repr(self)
