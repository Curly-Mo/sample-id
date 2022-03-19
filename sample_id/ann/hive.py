from __future__ import annotations

import abc
import datetime
import itertools
import json
import logging
import math
import operator
import os
import statistics
import tempfile
import zipfile
from typing import Any, Iterable, List, Optional, Sequence

from sample_id.fingerprint import Fingerprint

from . import Match, Matcher, MatcherMetadata, Neighbor

logger = logging.getLogger(__name__)


# TODO: Refactor, maybe to separate training from inference, because this should not be trainable
class HiveMatcher(Matcher):
    """A wrapper around a list of Matchers so that they act like a single Matcher (for inference only)."""

    def __init__(self, matchers: List[Matcher]):
        sr = next((matcher.meta.sr for matcher in matchers), None)
        hop_length = next((matcher.meta.hop_length for matcher in matchers), None)
        for matcher in matchers:
            if matcher.meta.sr != sr or matcher.meta.hop_length != hop_length:
                raise ValueError(f"Hive must all have the same sr and hop_length, can't add {matcher}")
        self.meta = MatcherMetadata(sr=sr, hop_length=hop_length)
        self.matchers = matchers

    def add_matcher(self, matcher: Matcher) -> HiveMatcher:
        if matcher.meta.sr != self.meta.sr or matcher.meta.hop_length != self.meta.hop_length:
            raise ValueError(f"Hive must all have the same sr and hop_length, can't add {matcher}")
        self.matchers.append(matcher)
        return self

    def init_model(self) -> Any:
        raise NotImplementedError(f"Don't do this.")

    def save_model(self, filepath: str, **kwargs) -> str:
        raise NotImplementedError(f"Don't do this.")

    def load_model(self, filepath: str, **kwargs) -> Any:
        raise NotImplementedError(f"Don't do this.")

    def add_fingerprint(self, fingerprint: Fingerprint, dedupe=True) -> Matcher:
        raise NotImplementedError(f"Don't do this.")

    def add_fingerprints(self, fingerprints: Iterable[Fingerprint], **kwargs) -> Matcher:
        raise NotImplementedError(f"Don't do this.")

    def can_add_fingerprint(self, fingerprint: Fingerprint) -> bool:
        return False

    def save(self, filepath: str, compress: bool = True, **kwargs) -> str:
        raise NotImplementedError(f"Don't do this, save the matchers individually.")

    @classmethod
    def create(cls, sr: Optional[int] = None, hop_length: Optional[int] = None, **kwargs) -> Matcher:
        raise NotImplementedError(f"Don't do this.")

    @classmethod
    def from_fingerprint(cls, fp: Fingerprint, **kwargs) -> Matcher:
        raise NotImplementedError(f"Don't do this.")

    @classmethod
    def from_fingerprints(cls, fingerprints: Sequence[Fingerprint], **kwargs) -> Matcher:
        raise NotImplementedError(f"Don't do this.")

    @classmethod
    def load(cls, filepaths: Iterable[str], **kwargs) -> Matcher:
        """Load multiple matchers from disk into hive."""
        matchers = []
        for filepath in filepaths:
            matcher = Matcher.load(filepath, **kwargs)
            matchers.append(matcher)
        return cls(matchers)

    def nearest_neighbors(self, fp: Fingerprint, k: int = 1) -> Sequence[Match]:
        hive_matches = []
        for matcher in self.matchers:
            matches = matcher.nearest_neighbors(fp, k)
            hive_matches.append(matches)
        resorted_matches = []
        for kp_hive_matches in zip(*hive_matches):
            top_k_neighbors = sorted(
                (neighbor for match in kp_hive_matches for neighbor in match.neighbors),
                key=operator.attrgetter("distance"),
            )[:k]
            head = next(match for match in kp_hive_matches)
            resorted_matches.append(Match(head.keypoint.kp, top_k_neighbors))
        return resorted_matches
