import logging
import os
from typing import Any, Dict, Iterable, Sequence

logger = logging.getLogger(__name__)


def class_repr(cls, filter_types: Sequence[Any] = [], **kwargs) -> str:
    attributes = class_attributes(cls, filter_types=filter_types)
    kwargstring = kv_string((kwargs, attributes))
    return f"{cls.__class__.__name__}({kwargstring})"


def kv_string(dicts: Iterable[Dict[Any, Any]]) -> str:
    return ",".join(f"{k}={v}" for d in dicts for k, v in d.items())


def class_attributes(cls, filter_types: Sequence[Any] = (int, float, bool, str)) -> Dict[str, Any]:
    return {
        k: v for k, v in vars(cls).items() if (type(v) in filter_types or not filter_types) and len(v.__repr__()) < 80
    }


def human_bytes(bytes: float) -> str:
    """Human readable string representation of bytes"""
    units = "bytes"
    if bytes > 1024:
        units = "KiB"
        bytes = bytes / 1024
    if bytes > 1024:
        units = "MiB"
        bytes = bytes / 1024
    if bytes > 1024:
        units = "GiB"
        bytes = bytes / 1024
    return f"%.1f {units}" % bytes


def filesize(filename: str) -> str:
    """Human readable string representation of filesize"""
    if not os.path.exists(filename):
        logger.warn(f"File {filename} does not exist")
        return human_bytes(0)
    return human_bytes(os.path.getsize(filename))
