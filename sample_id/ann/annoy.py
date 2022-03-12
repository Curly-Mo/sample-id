import logging
from typing import Any, Sequence

import annoy

from sample_id.fingerprint import Fingerprint

from . import Match, Matcher, MatcherMetadata, Neighbor

logger = logging.getLogger(__name__)


class AnnoyMatcher(Matcher):
    """Nearest neighbor matcher using annoy."""

    def __init__(self, metadata: MatcherMetadata):
        metadata.metric = vars(metadata).get("metric", "angular")
        metadata.n_features = vars(metadata).get("n_features", 128)
        metadata.n_trees = vars(metadata).get("n_trees", 40)
        metadata.n_jobs = vars(metadata).get("n_jobs", -1)
        super().__init__(metadata)
        self.on_disk = None
        self.built = False

    def init_model(self) -> Any:
        logger.info(f"Initializing Annoy Index with {self.meta}...")
        return annoy.AnnoyIndex(self.meta.n_features, metric=self.meta.metric)

    def save_model(self, filepath: str, prefault: bool = False) -> str:
        if not self.built:
            self.build()
        else:
            logger.info(f"Annoy Index already built.")
        if self.on_disk:
            logger.info(f"Annoy index already built_on_disk at {self.on_disk}.")
            return self.on_disk
        logger.info(f"Saving matcher model to {filepath}...")
        self.model.save(filepath, prefault=prefault)
        return filepath

    def load_model(self, filepath: str, prefault: bool = False) -> None:
        logger.info(f"Loading Annoy Index from {filepath}...")
        self.model.load(filepath, prefault=prefault)
        self.built = True
        return self.model

    def build(self) -> None:
        logger.info(f"Building Annoy Index with {self.meta}...")
        self.model.build(self.meta.n_trees, self.meta.n_jobs)
        self.built = True

    def on_disk_build(self, filename: str) -> None:
        logger.info(f"Building Annoy Index straight to disk: {filename}...")
        self.model.on_disk_build(filename)
        self.on_disk = filename

    def nearest_neighbors(self, fp: Fingerprint, k: int = 1) -> Sequence[Match]:
        matches = []
        for kp, desc in zip(fp.keypoints, fp.descriptors):
            indices, distances = self.model.get_nns_by_vector(desc, k, include_distances=True)
            kp_neighbors = [Neighbor(index, distance, self.meta) for index, distance in zip(indices, distances)]
            matches.append(Match(kp, kp_neighbors))
        return matches
