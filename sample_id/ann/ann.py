from __future__ import annotations

import abc
import datetime
import itertools
import json
import logging
import math
import os
import statistics
import tempfile
import zipfile
from collections import defaultdict
from typing import Any, Iterable, List, Optional, Sequence

import numpy as np

from sample_id import util
from sample_id.fingerprint import Fingerprint

from . import hough

logger = logging.getLogger(__name__)


MATCHER_FILENAME: str = "matcher.ann"
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
    def nearest_neighbors(self, fp: Fingerprint, k: int = 1) -> Iterable[Match]:
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

    def save(self, filepath: str, compress: bool = True, **kwargs) -> str:
        """Save this matcher to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger.info(f"Saving {self} to temporary dir: {tmpdir}")
            tmp_model_path = os.path.join(tmpdir, MATCHER_FILENAME)
            tmp_meta_path = os.path.join(tmpdir, META_FILENAME)
            tmp_model_path = self.save_model(tmp_model_path, **kwargs)
            self.meta.save(tmp_meta_path, compress=compress)
            logger.info(f"Model file {tmp_model_path} size: {util.filesize(tmp_model_path)}")
            logger.info(f"Metadata file {tmp_meta_path} size: {util.filesize(tmp_meta_path)}")
            with zipfile.ZipFile(filepath, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
                logger.info(f"Zipping {tmp_model_path} and {tmp_meta_path} into {zipf.filename}")
                zipf.write(tmp_model_path, arcname=MATCHER_FILENAME)
                zipf.write(tmp_meta_path, arcname=META_FILENAME)
        logger.info(f"Zipped file {filepath} size: {util.filesize(filepath)}")
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
    def load(cls, filepath: str, **kwargs) -> Matcher:
        """Load a matcher from disk."""
        with zipfile.ZipFile(filepath, "r") as zipf:
            tempdir = tempfile.TemporaryDirectory()
            tmp_model_path = os.path.join(tempdir.name, MATCHER_FILENAME)
            tmp_meta_path = os.path.join(tempdir.name, META_FILENAME)
            logger.info(f"Extracting matcher model to {tmp_model_path}...")
            zipf.extract(MATCHER_FILENAME, tempdir.name)
            logger.info(f"Extracting matcher metadata to {tmp_meta_path}...")
            zipf.extract(META_FILENAME, tempdir.name)
            meta = MatcherMetadata.load(tmp_meta_path)
            matcher = cls(meta)
            matcher.tempdir = tempdir
            matcher.load_model(tmp_model_path, **kwargs)
            return matcher

    def __repr__(self):
        return f"{self.__class__.__name__}({self.meta})"

    def filter_matches(
        self,
        matches: List[Match],
        abs_thresh=None,
        ratio_thresh=None,
        cluster_dist=1.0,
        cluster_size=3,
        match_orientation=True,
        ordered=False,
    ) -> List[List[Match]]:
        cluster_sample_dist = int(cluster_dist * self.meta.sr / self.meta.hop_length)
        return filter_matches(
            matches,
            abs_thresh=abs_thresh,
            ratio_thresh=ratio_thresh,
            cluster_dist=cluster_sample_dist,
            cluster_size=cluster_size,
            match_orientation=match_orientation,
            ordered=ordered,
        )

    def find_samples(
        self,
        fp: Fingerprint,
        abs_thresh=None,
        ratio_thresh=None,
        cluster_dist=1.0,
        cluster_size=3,
        match_orientation=True,
        ordered=False,
    ) -> Result:
        matches = self.nearest_neighbors(fp)
        clusters = self.filter_matches(
            matches,
            abs_thresh=abs_thresh,
            ratio_thresh=ratio_thresh,
            cluster_dist=cluster_dist,
            cluster_size=cluster_size,
            match_orientation=match_orientation,
            ordered=ordered,
        )
        return Result(fp, clusters)


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


class Match:
    def __init__(self, keypoint, neighbors: Sequence[Neighbor]):
        self.keypoint = Keypoint(keypoint)
        self.neighbors = neighbors

    def __repr__(self) -> str:
        return util.class_repr(self)


class Neighbor:
    def __init__(self, index: int, distance: float, meta: MatcherMetadata):
        self.index = index
        self.distance = distance
        self.keypoint = Keypoint(meta.index_to_kp[index])
        self.source_id = meta.index_to_id[index].item()

    def __repr__(self) -> str:
        return util.class_repr(self)


class Keypoint:
    def __init__(self, kp):
        self.x = kp[0].item()
        self.y = kp[1].item()
        self.scale = kp[2].item()
        self.orientation = kp[3].item()
        self.kp = kp

    def __repr__(self):
        return util.class_repr(self)


class Sample:
    def __init__(self, cluster: List[Match], sr, hop_length):
        deriv_min_x = min(match.keypoint.x for match in cluster)
        deriv_max_x = max(match.keypoint.x for match in cluster)
        source_min_x = min(match.neighbors[0].keypoint.x for match in cluster)
        source_max_x = max(match.neighbors[0].keypoint.x for match in cluster)
        derivative_start_seconds = deriv_min_x * hop_length / sr
        derivative_start_time = datetime.timedelta(seconds=derivative_start_seconds)
        derivative_end_seconds = deriv_max_x * hop_length / sr
        derivative_end_time = datetime.timedelta(seconds=derivative_end_seconds)
        source_start_seconds = source_min_x * hop_length / sr
        source_start_time = datetime.timedelta(seconds=source_start_seconds)
        source_end_seconds = source_max_x * hop_length / sr
        source_end_time = datetime.timedelta(seconds=source_end_seconds)

        combos = itertools.combinations(cluster, 2)
        stretch_factors = [
            abs(m2.keypoint.x - m1.keypoint.x) / abs(m2.neighbors[0].keypoint.x - m1.neighbors[0].keypoint.x)
            for m1, m2 in combos
        ]
        # TODO: read octave_bins from matcher somehow
        octave_bins = 24
        pitch_factors = [(m.neighbors[0].keypoint.y - m.keypoint.y) * 12 / octave_bins for m in cluster]

        self.derivative_start = derivative_start_time
        self.derivative_end = derivative_end_time
        self.source_start = source_start_time
        self.source_end = source_end_time
        self.pitch_shift = statistics.median(pitch_factors)
        self.time_stretch = statistics.median(stretch_factors)
        self.confidence = self.score(cluster)
        # TODO: for debugging purposes only
        self.cluster = cluster

    # TODO: do something not dumb here
    def score(self, cluster: List[Match]) -> float:
        sigmoid = lambda x: 1 / (1 + math.exp(-x))
        return sigmoid(len(cluster) / 100.0)

    def as_dict(self) -> dict:
        d = {k: str(v) if type(v) == datetime.timedelta else v for k, v in util.class_attributes(self, []).items()}
        d.pop("cluster", None)
        d
        return d

    def __repr__(self):
        return util.class_repr(self)


class Result:
    def __init__(self, fp: Fingerprint, clusters: List[List[Match]]):
        self.id = fp.id
        self.sources = defaultdict(list)
        for cluster in clusters:
            head = next(m for m in cluster)
            key = head.neighbors[0].source_id
            sample = Sample(cluster, fp.sr, fp.hop_length)
            self.sources[key].append(sample)

    def as_dict(self) -> dict:
        sources = [
            {"source": source, "samples": [sample.as_dict() for sample in samples]}
            for source, samples in self.sources.items()
        ]
        return {"id": self.id, "sources": sources}

    def __repr__(self):
        return util.class_repr(self)


def filter_matches(
    matches: List[Match],
    abs_thresh=None,
    ratio_thresh=None,
    cluster_dist=1,
    cluster_size=3,
    match_orientation=True,
    ordered=False,
):
    logger.info("Filtering nearest neighbors down to actual matched samples")
    if match_orientation:
        # Remove matches with differing orientations
        total = len(matches)
        for match in list(matches):
            orient = match.keypoint.orientation
            while match.neighbors and abs(orient - match.neighbors[0].keypoint.orientation) > 0.2:
                match.neighbors = match.neighbors[1:]
            if not match.neighbors:
                matches.remove(match)
            # elif len(match.neighbors) < 2:
            #     # logger.warn('Orientation check left < 2 neighbors')
            #     matches.remove(match)
        logger.info("Differing orientations removed: {}, remaining: {}".format(total - len(matches), len(matches)))
    if abs_thresh:
        # Apply absolute threshold
        total = len(matches)
        matches = [match for match in matches if match.neighbors[0].distance < abs_thresh]
        logger.info("Absolute threshold removed: {}, remaining: {}".format(total - len(matches), len(matches)))
    if ratio_thresh:
        # Apply ratio test
        total = len(matches)
        for match in list(matches):
            n1 = match.neighbors[0]
            n2 = next((n for n in match.neighbors if n.source_id != n1.source_id), None)
            if n2 is None:
                logger.warn("No second neighbor for ratio test, consider increasing k")
                d2 = n1.distance * 2
            else:
                d2 = n2.distance
            if not (n1.distance < ratio_thresh * d2):
                matches.remove(match)
        logger.info("Ratio threshold removed: {}, remaining: {}".format(total - len(matches), len(matches)))
    # Only keep when there are multiple within a time cluster
    # clusters = list(cluster_matches(matches, cluster_dist))
    # filtered_clusters = [cluster for cluster in clusters if len(cluster) >= cluster_size]
    filtered_clusters, clusters = hough.cluster(matches, cluster_size, cluster_dist)
    logger.info("Total Clusters: {}, filtered clusters: {}".format(len(clusters), len(filtered_clusters)))
    if ordered:
        orderedx_clusters = []
        ordered_clusters = []
        for cluster in filtered_clusters:
            sorted_trainx = sorted(cluster, key=lambda m: m.neighbors[0].keypoint.x)
            sorted_queryx = sorted(cluster, key=lambda m: m.keypoint.x)
            if sorted_trainx == sorted_queryx:
                orderedx_clusters.append(cluster)
        logger.info("Total Clusters: {}, orderedx clusters: {}".format(len(clusters), len(orderedx_clusters)))
        for cluster in orderedx_clusters:
            sorted_trainy = sorted(cluster, key=lambda m: m.neighbors[0].keypoint.y)
            sorted_queryy = sorted(cluster, key=lambda m: m.keypoint.y)
            if sorted_trainy == sorted_queryy:
                ordered_clusters.append(cluster)
        logger.info("Total Clusters: {}, ordered clusters: {}".format(len(clusters), len(ordered_clusters)))
        filtered_clusters = ordered_clusters
    matches = [match for cluster in filtered_clusters for match in cluster]
    logger.info("Filtered matches: {}".format(len(matches)))
    return filtered_clusters


def cluster_matches(matches, cluster_dist):
    class Cluster(object):
        def __init__(self, match):
            self.min_query = match.keypoint.x
            self.max_query = match.keypoint.x
            self.min_train = match.neighbors[0].keypoint.x
            self.max_train = match.neighbors[0].keypoint.x
            self.matches = [match]

        def add(self, match):
            if match.keypoint.x > self.min_query:
                self.min_query = match.keypoint.x
            if match.keypoint.x > self.max_query:
                self.max_query = match.keypoint.x
            if match.neighbors[0].keypoint.x < self.min_train:
                self.min_train = match.neighbors[0].keypoint.x
            if match.neighbors[0].keypoint.x > self.max_train:
                self.max_train = match.neighbors[0].keypoint.x
            self.matches.append(match)

        def merge(self, cluster):
            if cluster.min_query < self.min_query:
                self.min_query = cluster.min_query
            if cluster.max_query > self.max_query:
                self.max_query = cluster.max_query
            if cluster.min_train < self.min_train:
                self.min_train = cluster.min_train
            if cluster.max_train > self.max_train:
                self.max_train = cluster.max_train
            self.matches.extend(cluster.matches)

    logger.info("Clustering matches...")
    logger.info(f"cluster_dist: {cluster_dist}")
    matches = sorted(matches, key=lambda m: (m.neighbors[0].keypoint.source, m.keypoint.x))
    clusters = {}
    for source, group in itertools.groupby(matches, lambda m: m.neighbors[0].keypoint.source):
        for match in group:
            cluster_found = False
            for cluster in clusters.get(source, []):
                if (
                    match.keypoint.x >= cluster.min_query - cluster_dist
                    and match.keypoint.x <= cluster.max_query + cluster_dist
                ) and (
                    match.neighbors[0].keypoint.x >= cluster.min_train - cluster_dist
                    and match.neighbors[0].keypoint.x <= cluster.max_train + cluster_dist
                ):
                    if not any(
                        match.neighbors[0].keypoint.x == c.neighbors[0].keypoint.x
                        and match.neighbors[0].keypoint.y == c.neighbors[0].keypoint.y
                        for c in cluster.matches
                    ):
                        cluster_found = True
                        cluster.add(match)
            if not cluster_found:
                clusters.setdefault(source, []).append(Cluster(match))
        # Merge nearby clusters
        merged_clusters = clusters.get(source, [])
        for cluster in clusters.get(source, []):
            for c in merged_clusters:
                if (
                    c != cluster
                    and (
                        cluster.min_query >= c.min_query - cluster_dist
                        and cluster.max_query <= c.max_query + cluster_dist
                    )
                    and (
                        cluster.min_train >= c.min_train - cluster_dist
                        and cluster.max_train <= c.max_train + cluster_dist
                    )
                ):
                    cluster_points = set(
                        (m.neighbors[0].keypoint.x, m.neighbors[0].keypoint.y) for m in cluster.matches
                    )
                    c_points = set((m.neighbors[0].keypoint.x, m.neighbors[0].keypoint.y) for m in c.matches)
                    if cluster_points & c_points:
                        break
                    c.merge(cluster)
                    logging.info(len(merged_clusters))
                    merged_clusters.remove(cluster)
                    logging.info(len(merged_clusters))
                    cluster = c
        clusters[source] = merged_clusters
    clusters = [cluster.matches for sources in clusters.values() for cluster in sources]
    return clusters
