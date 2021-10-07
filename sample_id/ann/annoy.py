import logging
from typing import Any, Iterable, Optional

import annoy

from . import Matcher, MatcherMetadata

logger = logging.getLogger(__name__)


class AnnoyMatcher(Matcher):
    """Nearest neighbor matcher using annoy."""

    def __init__(self, metadata: MatcherMetadata):
        metadata.metric = vars(metadata).get("metric", "euclidean")
        metadata.n_features = vars(metadata).get("n_features", 128)
        metadata.n_trees = vars(metadata).get("n_trees", -1)
        metadata.n_jobs = vars(metadata).get("n_jobs", -1)
        super().__init__(metadata)
        self.on_disk = None
        self.built = False

    def init_model(self) -> Any:
        logger.info(f"Initializing Annoy Index with {self.meta}...")
        return annoy.AnnoyIndex(self.meta.n_features, metric=self.meta.metric)

    def save_model(self, filepath: str) -> str:
        if not self.built:
            self.build()
        if self.on_disk:
            logger.info(f"Annoy index already built on disk at {self.on_disk}.")
            return self.on_disk
        logger.info(f"Saving matcher model to {filepath}...")
        self.model.save(filepath)
        return filepath

    def load_model(self, filepath: str) -> None:
        logger.info(f"Loading Annoy Index from {filepath}...")
        self.model.load(filepath)
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
