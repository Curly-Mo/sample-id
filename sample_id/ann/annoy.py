import logging
from typing import Any, Iterable, Optional

import annoy

from . import Matcher, MatcherMetadata

logger = logging.getLogger(__name__)


class AnnoyMatcher(Matcher):
    """Nearest neighbor matcher using annoy."""

    def __init__(self, metadata: MatcherMetadata):
        metadata.metric = metadata.metric or "euclidean"
        metadata.n_features = metadata.n_features or 128
        metadata.n_trees = vars(metadata).get("n_trees", -1)
        metadata.n_jobs = vars(metadata).get("n_jobs", -1)
        super().__init__(metadata)
        self.on_disk = None

    def init_model(self) -> Any:
        return annoy.AnnoyIndex(self.meta.n_features, metric=self.meta.metric)

    def build(self) -> None:
        logger.info(f"Building Annoy Index with {self.meta}...")
        self.model.build(self.meta.n_trees, self.meta.n_jobs)

    def on_disk_build(self, filename: str) -> None:
        logger.info(f"Building Annoy Index straight to disk: {filename}...")
        self.model.on_disk_build(filename)
        self.on_disk = filename

    def save_model(self, filepath: str) -> str:
        self.build()
        if self.on_disk:
            logger.info(f"Annoy index already built on disk at {self.on_disk}.")
            return self.on_disk
        logger.info(f"Saving matcher model to {tmp_model_path}.")
        self.model.save(filepath)
        return filepath

    def load_model(self, filepath: str) -> None:
        logger.info(f"Loading Annoy Index from {filepath}...")
        self.model.load(filepath)
        return self.model
