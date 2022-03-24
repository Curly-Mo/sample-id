from __future__ import annotations

import bisect
import dataclasses
import datetime
import itertools
import logging
import math
import statistics
from collections import defaultdict
from dataclasses import InitVar, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from sample_id.fingerprint import Fingerprint, Keypoint

from . import ann, hough

logger = logging.getLogger(__name__)


@dataclass
class Match:
    """A match between a Keypoint and it's neighbors."""

    keypoint: Keypoint
    neighbors: Sequence[Neighbor]

    def __hash__(self):
        return hash((self.keypoint, next(iter(self.neighbors), None)))


@dataclass(unsafe_hash=True)
class Neighbor:
    """A nearest_neighbor queried from a Matcher."""

    index: int
    distance: float
    keypoint: Keypoint = field(init=False)
    source_id: str = field(init=False)
    meta: InitVar[ann.MatcherMetadata]

    def __post_init__(self, meta: ann.MatcherMetadata):
        self.keypoint = Keypoint(meta.index_to_kp[self.index])
        self.source_id = meta.index_to_id[self.index].item()


@dataclass
class Cluster:
    """A group of Matches."""

    matches: List[Match]

    def __iter__(self):
        return iter(self.matches)

    def __len__(self):
        return len(self.matches)


@dataclass
class Sample:
    derivative_start: datetime.timedelta
    derivative_end: datetime.timedelta
    source_start: datetime.timedelta
    source_end: datetime.timedelta
    pitch_shift_factor: Optional[float]
    time_stretch_factor: Optional[float]
    confidence: float
    size: int
    min_distance: float
    average_distance: float

    @classmethod
    def from_cluster(cls, cluster: Cluster, sr: int, hop_length: int) -> Sample:
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
        octave_bins = 36
        pitch_factors = [(m.neighbors[0].keypoint.y - m.keypoint.y) * 2 * 12 / octave_bins for m in cluster]

        pitch_shift = None if not pitch_factors else statistics.median(pitch_factors)
        time_stretch = None if not stretch_factors else statistics.median(stretch_factors)
        distances = [match.neighbors[0].distance for match in cluster]
        sample = cls(
            derivative_start_time,
            derivative_end_time,
            source_start_time,
            source_end_time,
            pitch_shift,
            time_stretch,
            cls.score(cluster),
            len(cluster),
            min(distances),
            statistics.mean(distances),
        )
        # TODO: for debugging purposes only
        sample.cluster = cluster
        return sample

    # TODO: do something not dumb here
    @classmethod
    def score(cls, cluster: Cluster) -> float:
        sigmoid = lambda x: 1.0 / (1 + math.exp(-x))
        distances = [match.neighbors[0].distance for match in cluster]
        return sigmoid(len(cluster) - 3) * (1 - statistics.mean(distances))
        # return sigmoid(len(cluster) - 3) * sigmoid(12 - abs(pitch_shift)) * sigmoid(1 - abs(time_stretch))

    def as_dict(self) -> dict:
        return {k: str(v) if type(v) == datetime.timedelta else v for k, v in dataclasses.asdict(self).items()}

    def __lt__(self, other: Sample) -> bool:
        """Default sort by confidence score"""
        return self.confidence < other.confidence

    @classmethod
    def basic_filter(cls, min_size: int = 2, min_distance: float = 0.2) -> Callable[[Sample], bool]:
        """Filter function keeping only samples with either a minimum size or distance."""

        def fn(sample: Sample) -> bool:
            return sample.size >= min_size or sample.min_distance <= min_distance

        return fn


@dataclass
class Result:
    id: str = field(init=False)
    sources: Dict[str, Any] = field(init=False)
    fp: InitVar[Fingerprint]
    clusters: InitVar[List[Cluster]]

    def __post_init__(self, fp: Fingerprint, clusters: List[Cluster]):
        self.id = fp.id
        self.sources = defaultdict(list)
        for cluster in clusters:
            head = next(m for m in cluster)
            key = head.neighbors[0].source_id
            sample = Sample.from_cluster(cluster, fp.sr, fp.hop_length)
            # keep samples sorted by confidence
            bisect.insort(self.sources[key], sample)

    def filter(self, sample_filter: Callable[[Sample], bool] = Sample.basic_filter()) -> Result:
        """Filter keeping only samples that fit the filter function."""
        sources = {source: sample_filter(samples) for source, samples in self.sources.items()}
        self.sources = {source: samples for source, samples in sources.items() if samples}
        return self

    def filter_min_size_or_distance(self, min_size: int = 2, min_distance: float = 0.2) -> Result:
        """Filter keeping only samples with either a minimum size or distance."""
        return self.filter(Sample.basic_filter(min_size=min_size, min_distance=min_distance))

    def as_dict(self, id_mapper: Callable[[str], str] = lambda i: i) -> dict:
        # Sort sources by max confidence score
        sources = sorted(
            [
                {"source": id_mapper(source), "samples": list(reversed([sample.as_dict() for sample in samples]))}
                for source, samples in self.sources.items()
            ],
            key=lambda source_d: max(sample.get("confidence", 0) for sample in source_d.get("samples", [])),
            reverse=True,
        )
        return {"id": id_mapper(self.id), "sources": sources}


def basic_cluster_filter(min_cluster_size: int = 2, min_distance: float = 0.2) -> Callable[[Cluster], bool]:
    """Filter function keeping only clusters with either a minimum size or distance"""

    def fn(cluster: Cluster) -> bool:
        return len(cluster) >= min_cluster_size or any(
            n.distance <= min_distance for match in cluster for n in match.neighbors
        )

    return fn


def filter_matches(
    matches: List[Match],
    abs_thresh: Optional[float] = 0.25,
    ratio_thresh: Optional[float] = None,
    cluster_dist: float = 20.0,
    cluster_size: int = 2,
    match_orientation: bool = True,
    ordered: bool = False,
    cluster_filter: Optional[Callable[[Cluster], bool]] = basic_cluster_filter(),
) -> List[Cluster]:
    logger.info("Filtering nearest neighbors down to actual matched samples")
    if match_orientation:
        # Remove matches with differing orientations
        total = len(matches)
        for match in list(matches):
            orient = match.keypoint.orientation
            while match.neighbors and abs(orient - match.neighbors[0].keypoint.orientation) > 0.2:
                # replace(match, neighbors=match.neighbors[1:])
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
    if cluster_filter:
        filtered = [cluster for cluster in filtered_clusters if cluster_filter(cluster)]
        logger.info(
            f"Custom filter removed {len(filtered_clusters) - len(filtered)}, {len(filtered)} clusters remaining."
        )
        filtered_clusters = filtered
    matches = [match for cluster in filtered_clusters for match in cluster]
    logger.info(f"Filtered matches: {len(matches)}")
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
    matches = sorted(matches, key=lambda m: (m.neighbors[0].source_id, m.keypoint.x))
    clusters = {}
    for source, group in itertools.groupby(matches, lambda m: m.neighbors[0].source_id):
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
