"""Download and safely extract VneuroTK sample datasets.

Usage
-----
>>> from vneurotk.datasets import sample
>>> root = sample.data_path("nod-meg")          # download only NOD-MEG
>>> root = sample.data_path("monkey-vision")    # download only MonkeyVision
>>> root = sample.data_path()                   # download all datasets

Directory layout after download
--------------------------------
``data_path()`` returns the extracted root that contains the selected sub-trees::

    <root>/
    ├── nod-meg/
    │   ├── meg/
    │   │   └── sub-01_ses-ImageNet01_task-ImageNet_run-01_meg_clean.fif
    │   ├── events/
    │   │   └── sub-01_events.csv
    │   └── stimuli/
    │       └── <image_id>.JPEG  (200 images from run 01)
    └── monkey-vision/
        └── sessions/
            └── 251024_FanFan_nsd1w_MSB/
                ├── TrialRaster_251024_FanFan_nsd1w_MSB.h5
                ├── TrialRecord_251024_FanFan_nsd1w_MSB.csv
                ├── MeanFr_251024_FanFan_nsd1w_MSB.h5
                ├── ChMeanFr_251024_FanFan_nsd1w_MSB.h5
                ├── ChStimFr_251024_FanFan_nsd1w_MSB.h5
                ├── ChTrialRaster_251024_FanFan_nsd1w_MSB.h5
                ├── ChTrialRecord_251024_FanFan_nsd1w_MSB.csv
                ├── UnitProp_251024_FanFan_nsd1w_MSB.csv
                └── ChProp_251024_FanFan_nsd1w_MSB.csv
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import stat
import tempfile
import time
import uuid
import zipfile
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, BinaryIO, Literal, cast

import pooch

# Zenodo configuration and verified archive metadata.
_ZENODO_RECORD = "20094167"
_BASE_URL = f"https://zenodo.org/records/{_ZENODO_RECORD}/files/"

_NOD_FNAME = "vneurotk-nod-meg-sample.zip"
_NOD_HASH = "sha256:cebcec0bab57d548c486d7e4456c1e56c5832a47e4793ddb1e8f5f6a4d403968"
_NOD_SIZE = 87_243_257
_NOD_MAX_SIZE = _NOD_SIZE

_MV_FNAME = "vneurotk-monkey-vision-sample.zip"
_MV_HASH = "sha256:bb5c1a8aab1faa4fb97d89adaabfa301bea7f4879b54b102f3b3a17f35ada94e"
_MV_SIZE = 191_862_733
_MV_MAX_SIZE = _MV_SIZE

_EXTRACT_DIR = "vneurotk-samples"
_HTTP_TIMEOUT = (10, 60)  # requests connect and read timeouts, in seconds
_DOWNLOAD_RETRIES = 2  # two retries after the initial attempt

# Extraction limits are deliberately well above the two verified samples.
_MAX_ARCHIVE_MEMBERS = 10_000
_MAX_MEMBER_SIZE = 2 * 1024**3
_MAX_TOTAL_SIZE = 10 * 1024**3
_MAX_COMPRESSION_RATIO = 1_000
_COPY_CHUNK_SIZE = 1024 * 1024

# A dataset extraction transaction waits at most this long for another process.
_EXTRACTION_LOCK_TIMEOUT = 120.0
_EXTRACTION_LOCK_STALE_AGE = 60.0
_LOCK_INITIAL_BACKOFF = 0.05
_LOCK_MAX_BACKOFF = 1.0

# Mapping from user-facing dataset name to
# (filename, digest, expected archive size, maximum archive size).
_DATASETS: dict[str, tuple[str, str, int, int]] = {
    "nod-meg": (_NOD_FNAME, _NOD_HASH, _NOD_SIZE, _NOD_MAX_SIZE),
    "monkey-vision": (_MV_FNAME, _MV_HASH, _MV_SIZE, _MV_MAX_SIZE),
}

# NOD-MEG metadata constants.
NOD_SUBJECT = "01"
NOD_SESSION = "ImageNet01"
NOD_TASK = "ImageNet"
NOD_RUN = "01"

# MonkeyVision metadata constants.
EPHYS_SESSION_ID = "251024_FanFan_nsd1w_MSB"


class _BoundedWriter:
    """File proxy that refuses to write beyond an archive's byte limit."""

    def __init__(self, output_file: BinaryIO, max_size: int):
        self._output_file = output_file
        self._max_size = max_size
        self.written = 0

    def write(self, data: bytes) -> int:
        size = len(data)
        if self.written + size > self._max_size:
            raise ValueError(f"Sample archive download exceeds the {self._max_size:,}-byte limit")
        result = self._output_file.write(data)
        self.written += result
        return result

    def flush(self) -> None:
        self._output_file.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._output_file, name)


class _BoundedHTTPDownloader:
    """Pooch-compatible size guard around its requests HTTP downloader."""

    def __init__(
        self,
        *,
        expected_size: int,
        max_size: int,
        downloader: Callable[..., Any] | None = None,
        progressbar: bool = True,
        timeout: tuple[int, int] = _HTTP_TIMEOUT,
    ):
        if expected_size < 0 or max_size < expected_size:
            raise ValueError("archive byte bounds must satisfy 0 <= expected_size <= max_size")
        self.expected_size = expected_size
        self.max_size = max_size
        self.downloader = (
            pooch.HTTPDownloader(progressbar=progressbar, timeout=timeout) if downloader is None else downloader
        )

    def __call__(
        self,
        url: str,
        output_file: str | os.PathLike[str] | BinaryIO,
        pooch_instance: pooch.Pooch | None,
        check_only: bool = False,
    ) -> bool | None:
        if check_only:
            return self.downloader(url, output_file, pooch_instance, check_only=True)

        is_path = not hasattr(output_file, "write")
        opened_file: BinaryIO | None = None
        if is_path:
            opened_file = Path(cast(str | os.PathLike[str], output_file)).open("w+b")
            target = opened_file
        else:
            target = cast(BinaryIO, output_file)
        try:
            bounded = _BoundedWriter(target, self.max_size)
            self.downloader(url, bounded, pooch_instance)
            if bounded.written != self.expected_size:
                raise ValueError(
                    f"Sample archive download has {bounded.written:,} bytes; expected {self.expected_size:,}"
                )
        finally:
            if opened_file is not None:
                opened_file.close()
        return None


def _selection(dataset: str | list[str] | None) -> list[str]:
    """Validate and normalize a public dataset selection."""
    valid = tuple(_DATASETS)
    if dataset is None:
        return list(valid)
    if isinstance(dataset, str):
        if not dataset:
            raise ValueError("dataset must not be an empty string")
        names = [dataset]
    elif isinstance(dataset, list):
        if not dataset:
            raise ValueError("dataset list must not be empty")
        if any(not isinstance(name, str) for name in dataset):
            raise TypeError("every dataset name must be a string")
        if any(not name for name in dataset):
            raise ValueError("dataset names must not be empty strings")
        names = list(dataset)
    else:
        raise TypeError("dataset must be a string, a list of strings, or None")

    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate dataset name(s) {duplicates}.")
    unknown = sorted(set(names).difference(valid))
    if unknown:
        raise ValueError(f"Unknown dataset(s) {unknown}. Choose from {sorted(valid)}.")
    return names


def _archive_member_path(info: zipfile.ZipInfo, dataset: str) -> PurePosixPath:
    """Return a validated POSIX member path rooted in *dataset*."""
    name = info.filename
    path = PurePosixPath(name)
    windows_path = PureWindowsPath(name)
    if not name or "\\" in name or path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise ValueError(f"Unsafe archive member path: {name!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe archive member path: {name!r}")
    if not path.parts or path.parts[0] != dataset:
        raise ValueError(f"Archive member is outside the {dataset!r} dataset root: {name!r}")

    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise ValueError(f"Archive member is a symbolic link: {name!r}")
    file_type = stat.S_IFMT(mode)
    if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise ValueError(f"Archive member has an unsupported file type: {name!r}")
    return path


def _validate_archive(archive: zipfile.ZipFile, dataset: str) -> list[tuple[zipfile.ZipInfo, PurePosixPath]]:
    """Validate all members before writing any extracted data."""
    members = archive.infolist()
    if not members:
        raise ValueError("Sample archive is empty")
    if len(members) > _MAX_ARCHIVE_MEMBERS:
        raise ValueError(f"Sample archive has more than {_MAX_ARCHIVE_MEMBERS:,} members")

    validated: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
    seen: set[PurePosixPath] = set()
    total_size = 0
    for info in members:
        path = _archive_member_path(info, dataset)
        if path in seen:
            raise ValueError(f"Sample archive contains a duplicate member: {info.filename!r}")
        seen.add(path)
        if info.file_size > _MAX_MEMBER_SIZE:
            raise ValueError(f"Archive member exceeds the {_MAX_MEMBER_SIZE:,}-byte limit: {info.filename!r}")
        total_size += info.file_size
        if total_size > _MAX_TOTAL_SIZE:
            raise ValueError(f"Sample archive exceeds the {_MAX_TOTAL_SIZE:,}-byte uncompressed limit")
        if info.file_size and (info.compress_size == 0 or info.file_size / info.compress_size > _MAX_COMPRESSION_RATIO):
            raise ValueError(
                f"Archive member exceeds the {_MAX_COMPRESSION_RATIO:,}:1 compression-ratio limit: {info.filename!r}"
            )
        validated.append((info, path))
    return validated


def _extract_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    destination: Path,
    total_written: int,
) -> int:
    """Extract one member while independently enforcing declared limits."""
    if info.is_dir():
        destination.mkdir(parents=True, exist_ok=True)
        return total_written

    destination.parent.mkdir(parents=True, exist_ok=True)
    member_written = 0
    with archive.open(info) as source, destination.open("xb") as target:
        while chunk := source.read(_COPY_CHUNK_SIZE):
            member_written += len(chunk)
            total_written += len(chunk)
            if member_written > _MAX_MEMBER_SIZE or total_written > _MAX_TOTAL_SIZE:
                raise ValueError("Archive expanded beyond its declared extraction limits")
            target.write(chunk)
    return total_written


class _ExtractionLock(AbstractContextManager["_ExtractionLock"]):
    """Portable per-dataset lock with bounded wait and stale recovery.

    Acquisition waits for at most ``_EXTRACTION_LOCK_TIMEOUT`` seconds. A lock
    older than ``_EXTRACTION_LOCK_STALE_AGE`` is reclaimed only when its owner
    is known to be dead on this host, or when no usable owner metadata exists.
    """

    def __init__(self, path: Path):
        self.path = path
        self._fd: int | None = None
        self._token = uuid.uuid4().hex

    def __enter__(self) -> _ExtractionLock:
        deadline = time.monotonic() + _EXTRACTION_LOCK_TIMEOUT
        backoff = _LOCK_INITIAL_BACKOFF
        metadata = json.dumps(
            {
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "created": time.time(),
                "token": self._token,
            },
            separators=(",", ":"),
        ).encode("ascii")

        while True:
            try:
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                self._reclaim_if_stale()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timed out after {_EXTRACTION_LOCK_TIMEOUT:g} seconds waiting for "
                        f"sample extraction lock {self.path}"
                    ) from None
                time.sleep(min(backoff, remaining))
                backoff = min(backoff * 2, _LOCK_MAX_BACKOFF)
                continue

            self._fd = fd
            try:
                os.write(fd, metadata)
                os.fsync(fd)
            except BaseException:
                self._release()
                raise
            return self

    def __exit__(self, *exc_info: object) -> None:
        self._release()

    def _reclaim_if_stale(self) -> None:
        try:
            lock_stat = self.path.stat()
        except FileNotFoundError:
            return
        if time.time() - lock_stat.st_mtime <= _EXTRACTION_LOCK_STALE_AGE:
            return

        owner: dict[str, Any] = {}
        try:
            loaded = json.loads(self.path.read_text(encoding="ascii"))
            if isinstance(loaded, dict):
                owner = loaded
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass

        hostname = owner.get("hostname")
        if hostname and hostname != socket.gethostname():
            return
        if hostname == socket.gethostname():
            pid = owner.get("pid")
            if not isinstance(pid, int) or pid <= 0 or _pid_is_running(pid):
                return

        claim = self.path.with_name(f"{self.path.name}.reclaim-{os.getpid()}-{uuid.uuid4().hex}")
        try:
            os.link(self.path, claim)
            current_stat = self.path.stat()
            claim_stat = claim.stat()
            if (current_stat.st_dev, current_stat.st_ino) == (claim_stat.st_dev, claim_stat.st_ino):
                self.path.unlink()
        except (FileExistsError, FileNotFoundError, OSError):
            return
        finally:
            claim.unlink(missing_ok=True)

    def _release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            owned_stat = os.fstat(fd)
            try:
                current_stat = self.path.stat()
            except FileNotFoundError:
                return
            if (owned_stat.st_dev, owned_stat.st_ino) == (current_stat.st_dev, current_stat.st_ino):
                self.path.unlink(missing_ok=True)
        finally:
            os.close(fd)


def _pid_is_running(pid: int) -> bool:
    """Return whether *pid* is live on this host, conservatively on errors."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    return True


class _SafeZipProcessor:
    """Pooch processor that validates and atomically installs one dataset tree."""

    def __init__(self, dataset: str, digest: str):
        self.dataset = dataset
        self.digest = digest

    def __call__(
        self,
        fname: str,
        action: Literal["download", "update", "fetch"],
        pooch_instance: pooch.Pooch | None,
    ) -> str:
        del action
        archive_path = Path(fname)
        cache = pooch_instance.abspath if pooch_instance is not None else archive_path.parent
        extract_root = cache / _EXTRACT_DIR
        destination = extract_root / self.dataset
        marker = extract_root / f".{self.dataset}.complete"
        marker_value = f"{self.digest}\n"

        extract_root.mkdir(parents=True, exist_ok=True)
        lock = extract_root / f".{self.dataset}.lock"
        with _ExtractionLock(lock):
            if destination.is_dir() and marker.is_file():
                try:
                    if marker.read_text(encoding="ascii") == marker_value:
                        return str(destination)
                except OSError:
                    pass
            return self._extract_and_install(archive_path, extract_root, destination, marker, marker_value)

    def _extract_and_install(
        self,
        archive_path: Path,
        extract_root: Path,
        destination: Path,
        marker: Path,
        marker_value: str,
    ) -> str:
        """Extract and install while the caller holds the dataset lock."""
        staging_dir = Path(tempfile.mkdtemp(prefix=f".{self.dataset}-", dir=extract_root))
        staged_dataset = staging_dir / self.dataset
        backup: Path | None = None
        try:
            with zipfile.ZipFile(archive_path) as archive:
                members = _validate_archive(archive, self.dataset)
                total_written = 0
                for info, relative_path in members:
                    total_written = _extract_member(
                        archive,
                        info,
                        staging_dir.joinpath(*relative_path.parts),
                        total_written,
                    )
            if not staged_dataset.is_dir():
                raise ValueError(f"Sample archive does not contain a {self.dataset!r} dataset directory")

            if destination.exists():
                backup = Path(tempfile.mkdtemp(prefix=f".{self.dataset}-backup-", dir=extract_root))
                backup.rmdir()
                os.replace(destination, backup)

            marker_tmp: Path | None = None
            installed = False
            try:
                with tempfile.NamedTemporaryFile(
                    "w",
                    encoding="ascii",
                    prefix=f"{marker.name}.",
                    suffix=".tmp",
                    dir=extract_root,
                    delete=False,
                ) as marker_file:
                    marker_tmp = Path(marker_file.name)
                    marker_file.write(marker_value)
                    marker_file.flush()
                    os.fsync(marker_file.fileno())
                os.replace(staged_dataset, destination)
                installed = True
                os.replace(marker_tmp, marker)
            except BaseException:
                if installed:
                    if destination.is_dir():
                        shutil.rmtree(destination)
                    else:
                        destination.unlink(missing_ok=True)
                if backup is not None and backup.exists():
                    os.replace(backup, destination)
                raise
            finally:
                if marker_tmp is not None:
                    marker_tmp.unlink(missing_ok=True)
            if backup is not None:
                shutil.rmtree(backup, ignore_errors=True)
            return str(destination)
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)


def data_path(
    dataset: str | list[str] | None = None,
    path: str | Path | None = None,
    progressbar: bool = True,
) -> Path:
    """Return the root directory of VneuroTK sample datasets.

    Downloads and caches on first call; subsequent calls reuse verified archives
    and digest-bound completed extractions. Concurrent extraction calls for the
    same cache and dataset wait at most 120 seconds for the active transaction.
    The default cache location is ``~/.cache/vneurotk/`` (platform-specific).

    Zip files are fetched from Zenodo (DOI 10.5281/zenodo.20094167), verified
    with SHA256, validated, and extracted without ``extractall`` into a shared
    ``vneurotk-samples/`` sub-directory.

    Parameters
    ----------
    dataset : str or list of str or None
        Which dataset(s) to download. Accepted values are ``"nod-meg"``,
        ``"monkey-vision"``, a non-empty duplicate-free list of those names,
        or ``None`` (default) to download both.
    path : str or Path or None
        Explicit cache directory. Only ``None`` selects Pooch's platform cache.
    progressbar : bool
        Show a tqdm progress bar while downloading. Defaults to ``True``.

    Returns
    -------
    Path
        Absolute extracted root containing the selected dataset sub-trees.

    Examples
    --------
    >>> from vneurotk.datasets import sample
    >>> root = sample.data_path("nod-meg")
    >>> nod = root / "nod-meg"
    >>> nod.name
    'nod-meg'
    """
    names = _selection(dataset)
    cache = pooch.os_cache("vneurotk") if path is None else Path(path).expanduser()
    fetcher = pooch.create(
        path=cache,
        base_url=_BASE_URL,
        registry={fname: digest for fname, digest, _expected_size, _max_size in _DATASETS.values()},
        retry_if_failed=_DOWNLOAD_RETRIES,
    )

    for name in names:
        fname, digest, expected_size, max_size = _DATASETS[name]
        downloader = _BoundedHTTPDownloader(
            expected_size=expected_size,
            max_size=max_size,
            progressbar=progressbar,
            timeout=_HTTP_TIMEOUT,
        )
        fetcher.fetch(fname, downloader=downloader, processor=_SafeZipProcessor(name, digest))

    return fetcher.abspath / _EXTRACT_DIR
