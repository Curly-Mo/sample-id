from __future__ import annotations

import itertools
import logging
import math
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from . import query

logger = logging.getLogger(__name__)


def cluster(
    matches: List[query.Match], cluster_size: int = 3, cluster_dist: float = 20
) -> Tuple[List[query.Cluster], List[query.Cluster]]:
    logger.info("Clustering matches...")
    logger.debug(f"cluster_dist: {cluster_dist} samples")
    clusters = set()
    votes = ght(matches, cluster_dist)
    for source, bins in votes.items():
        source_clusters = set()
        for bin, cluster in bins.items():
            if len(cluster) >= cluster_size:
                source_clusters.add(query.Cluster(cluster))
        source_clusters = merge_nearby_clusters(source_clusters, cluster_dist)
        clusters = clusters.union(source_clusters)
    clusters = list(clusters)
    total_clusters = [query.Cluster(c) for bins in votes.values() for c in bins.values()]
    return clusters, total_clusters


def ght(
    matches: List[query.Match], cluster_dist: float = 20
) -> Dict[str, Dict[Tuple[float, float, float], Set[query.Match]]]:
    """Generalized Hough transform"""
    votes: Dict[str, Dict[Tuple[float, float, float], Set[query.Match]]] = defaultdict(lambda: defaultdict(set))
    try:
        dim = max(m.neighbors[0].keypoint.scale for m in matches)
    except:
        dim = 2
    for match in matches:
        ds = round_to(match.keypoint.scale / match.neighbors[0].keypoint.scale, 2)
        # d_theta = round_to(match.keypoint.orientation - match.neighbors[0].keypoint.orientation, 0.5)
        dx = round_to(match.keypoint.x - match.neighbors[0].keypoint.x, 1.5 * dim)
        dy = round_to(match.keypoint.y - match.neighbors[0].keypoint.y, 1.5 * dim)
        bins = itertools.product(*(dx, dy, ds))
        for bin in bins:
            x_vals = [m.neighbors[0].keypoint.x for m in votes[match.neighbors[0].source_id][bin]]
            try:
                min_x = min(x_vals)
                max_x = max(x_vals)
            except:
                min_x = max_x = match.neighbors[0].keypoint.x
            if min_x - cluster_dist < match.neighbors[0].keypoint.x < max_x + cluster_dist:
                votes[match.neighbors[0].source_id][bin].add(match)
    return votes


def round_to(x: float, base: float = 1, sig_figs: int = 4) -> Tuple[float, float]:
    lo = round(base * math.floor(float(x) / base), sig_figs)
    hi = round(base * math.ceil(float(x) / base), sig_figs)
    return (lo, hi)


def merge_nearby_clusters(clusters: Set[query.Cluster], cluster_dist: float) -> Set[query.Cluster]:
    # Merge nearby clusters
    merged_clusters = set()
    for cluster in clusters:
        merged = False
        for c in merged_clusters:
            if (
                c.min_deriv_x - cluster_dist <= cluster.min_deriv_x
                and cluster.max_deriv_x <= c.max_deriv_x + cluster_dist
            ) and (
                c.min_source_x - cluster_dist <= cluster.min_source_x
                and cluster.max_source_x <= c.max_source_x + cluster_dist
            ):
                c.merge(cluster)
                merged = True
        if not merged:
            merged_clusters.add(cluster)
    return merged_clusters
