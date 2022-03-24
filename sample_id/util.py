import functools
import logging
import os
import shutil
import tarfile
import tempfile
from typing import Any, Dict, Iterable, Optional, Sequence

import mgzip

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
    delete_added: bool = True,
) -> str:
    """Tar files."""
    with tarfile.open(output_filename, mode="w") as tarf:
        for file, arcname in zip(files, file_arcnames):
            tarf.add(file, arcname=arcname)
            if delete_added:
                os.remove(file)
    return output_filename


def untar(input_tarfile: str, members: Iterable[str], output_dir: str) -> Iterable[str]:
    """Untar an tarball."""
    output_filenames = []
    with tarfile.open(input_tarfile, mode="r") as tarf:
        for member in members:
            out_filename = os.path.join(output_dir, member)
            logger.info(f"Extracting {member} to {out_filename}...")
            tarf.extract(member, path=output_dir)
            output_filenames.append(out_filename)
    return output_filenames


def gzip_file(
    output_filename: str,
    input_filename: str,
    compress_level: int = 9,
    blocksize: int = 5 * 1024 * 1024,
    threads: Optional[int] = None,
) -> str:
    """Gzip a file using mgzip for multithreading."""
    with mgzip.open(
        output_filename, mode="wb", compresslevel=compress_level, blocksize=blocksize, thread=threads
    ) as f_out:
        with open(input_filename, "rb") as f_in:
            shutil.copyfileobj(f_in, f_out, length=blocksize // 2)
    return output_filename


def gunzip_file(
    input_filename: str,
    output_filename: str,
    blocksize: int = 5 * 1024 * 1024,
    threads: Optional[int] = None,
) -> str:
    """Gzip a file using mgzip for multithreading."""
    with open(output_filename, mode="wb") as f_out:
        with mgzip.open(input_filename, mode="rb", blocksize=blocksize, thread=threads) as f_in:
            shutil.copyfileobj(f_in, f_out, length=blocksize // 2)
    return output_filename
