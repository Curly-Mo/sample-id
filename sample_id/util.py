import functools
import logging
import os
import pathlib
import shutil
import tarfile
import tempfile
import weakref
from typing import Any, Dict, Iterable, Optional, Sequence, Union

import numpy as np
from pigz_python import pigz_python

logger = logging.getLogger(__name__)

COMPRESS_LEVEL_BEST = pigz_python._COMPRESS_LEVEL_BEST
DEFAULT_BLOCK_SIZE_KB = pigz_python.DEFAULT_BLOCK_SIZE_KB
CPU_COUNT = os.cpu_count()


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


def basic_attribute_repr(cls):
    @functools.wraps(cls, updated=())
    class ReprDecorated(cls):
        def __repr__(self) -> str:
            return class_repr(self)

    return ReprDecorated


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


def tar_files(
    output_filename: str,
    files: Iterable[str],
    file_arcnames: Iterable[str],
    compression: str = "gz",
    compresslevel=9,
    delete_added: bool = True,
) -> str:
    """Tar files."""
    kwargs = {"compresslevel": compresslevel} if compression else {}
    with tarfile.open(output_filename, mode=f"w:{compression}", **kwargs) as tarf:
        for file, arcname in zip(files, file_arcnames):
            tarf.add(file, arcname=arcname)
            if delete_added:
                os.remove(file)
    return output_filename


def untar_members(input_tarfile: str, members: Iterable[str], output_dir: str) -> Iterable[str]:
    """Extract files from a tarball."""
    output_filenames = []
    with tarfile.open(input_tarfile, mode="r") as tarf:
        for member in members:
            out_filename = os.path.join(output_dir, member)
            logger.info(f"Extracting {member} to {out_filename}...")
            tarf.extract(member, path=output_dir)
            output_filenames.append(out_filename)
    return output_filenames


def tar_gz_files(
    output_filename: str,
    files: Iterable[str],
    file_arcnames: Iterable[str],
    compression: str = "gz",
    compresslevel=COMPRESS_LEVEL_BEST,
    blocksize=DEFAULT_BLOCK_SIZE_KB,
    workers=CPU_COUNT,
    delete_added: bool = True,
) -> str:
    """Tar and gzip files using pigz for multithreading."""
    with tempfile.NamedTemporaryFile() as tarf:
        if compression != "gz":
            # Can't use pigz for the compression
            return tar_files(
                output_filename,
                files,
                file_arcnames,
                delete_added=delete_added,
                compression=compression,
                compresslevel=compresslevel,
            )
            tmp_file = tarf.name
        else:
            tar_files(tarf.name, files, file_arcnames, delete_added=delete_added, compression="")
            tmp_file = f"{tarf.name}.gz"
            pigz_python.compress_file(
                tarf.name,
                compresslevel=compresslevel,
                blocksize=blocksize,
                workers=workers,
            )

        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        shutil.move(tmp_file, output_filename)
    return output_filename


class NamedTemporaryDirectory(tempfile.TemporaryDirectory):
    """A wrapper around tempfile.TemporaryDirectory that allows for specifying the path."""

    def __init__(self, path: str):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        self.name = path
        self._finalizer = weakref.finalize(
            self, self._cleanup, self.name, warn_message="Implicitly cleaning up {!r}".format(self)
        )

    @classmethod
    def of(cls, dir: Union[str, tempfile.TemporaryDirectory, None]) -> tempfile.TemporaryDirectory:
        """Convenience method to get a TemporaryDirectory from a Union of possible dir references."""
        if isinstance(dir, tempfile.TemporaryDirectory):
            return dir
        if isinstance(dir, str):
            return cls(dir)
        return tempfile.TemporaryDirectory()


def round_array(array: np.ndarray, to_nearest: float = 10.0) -> np.ndarray:
    return (array / to_nearest).round() * to_nearest
