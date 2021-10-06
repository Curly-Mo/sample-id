import logging
from typing import Any, Iterable, Optional

import annoy

from . import Matcher, MatcherMetadata

logger = logging.getLogger(__name__)


class AnnoyMatcher(Matcher):
    """Nearest neighbor matcher using annoy."""

    def __init__(self, metadata: MatcherMetadata):
        metadata.metric = "euclidean"
        super().__init__(metadata)
        self.on_disk = None
        self.n_trees = -1
        self.n_jobs = -1

    def init_model(self) -> Any:
        return annoy.AnnoyIndex(self.meta.n_features, metric=self.meta.metric)

    def build(self, n_trees: int = -1, n_jobs: int = -1) -> None:
        logger.info("Building Annoy Index...")
        self.model.build(n_trees, n_jobs)

    def on_disk_build(self, filename: str) -> None:
        self.model.on_disk_build(filename)
        self.on_disk = filename

    def save_model(self, filepath: str) -> str:
        self.build(self.n_trees, self.n_jobs)
        if self.on_disk:
            return self.on_disk
        self.model.save(filepath)
        return filepath

    def load_model(self, filepath: str) -> None:
        logger.info(f"Loading Annoy Index from {filepath}...")
        self.model.load(filepath)
        return self.model
