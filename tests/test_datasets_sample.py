from __future__ import annotations

import hashlib
import io
import json
import multiprocessing
import os
import shutil
import socket
import stat
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pooch
import pytest

from vneurotk.datasets import sample


def _zip_bytes(members: dict[str, bytes], *, compression: int = zipfile.ZIP_DEFLATED) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=compression) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return output.getvalue()


def _write_zip(path: Path, members: dict[str, bytes]) -> str:
    content = _zip_bytes(members)
    path.write_bytes(content)
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _process_extract(archive: str, digest: str, start, results) -> None:
    """Run one extraction in a child process and return a serializable result."""
    start.wait()
    try:
        result = sample._SafeZipProcessor("nod-meg", digest)(archive, "fetch", None)
    except BaseException as error:
        results.put((False, type(error).__name__, str(error)))
    else:
        results.put((True, result))


@pytest.mark.parametrize(
    ("selection", "error", "match"),
    [
        ("", ValueError, "empty string"),
        ([], ValueError, "must not be empty"),
        (["nod-meg", "nod-meg"], ValueError, "Duplicate"),
        (["nod-meg", ""], ValueError, "empty strings"),
        (["nod-meg", 1], TypeError, "must be a string"),
        (("nod-meg",), TypeError, "string, a list"),
        (1, TypeError, "string, a list"),
        ("unknown", ValueError, "Unknown dataset"),
        (["nod-meg", "unknown"], ValueError, "Unknown dataset"),
    ],
)
def test_data_path_rejects_invalid_selection(selection, error, match):
    with pytest.raises(error, match=match):
        sample.data_path(selection, progressbar=False)


def test_data_path_orchestrates_pooch_with_explicit_path(monkeypatch, tmp_path):
    fetches: list[tuple[str, sample._BoundedHTTPDownloader, sample._SafeZipProcessor]] = []
    create_kwargs: dict[str, object] = {}

    class Fetcher:
        abspath = tmp_path.resolve()

        def fetch(self, fname, *, downloader, processor):
            fetches.append((fname, downloader, processor))

    def fake_create(**kwargs):
        create_kwargs.update(kwargs)
        return Fetcher()

    monkeypatch.setattr(sample.pooch, "create", fake_create)
    monkeypatch.setattr(sample.pooch, "os_cache", lambda _name: pytest.fail("path must be used explicitly"))

    root = sample.data_path(["monkey-vision", "nod-meg"], path=tmp_path, progressbar=False)

    assert root == tmp_path.resolve() / "vneurotk-samples"
    assert create_kwargs == {
        "path": tmp_path,
        "base_url": sample._BASE_URL,
        "registry": {
            sample._NOD_FNAME: sample._NOD_HASH,
            sample._MV_FNAME: sample._MV_HASH,
        },
        "retry_if_failed": 2,
    }
    assert [call[0] for call in fetches] == [sample._MV_FNAME, sample._NOD_FNAME]
    assert [call[1].expected_size for call in fetches] == [sample._MV_SIZE, sample._NOD_SIZE]
    assert [call[1].max_size for call in fetches] == [sample._MV_MAX_SIZE, sample._NOD_MAX_SIZE]
    assert all(isinstance(call[1].downloader, pooch.HTTPDownloader) for call in fetches)
    assert all(
        call[1].downloader.kwargs["timeout"] == (10, 60)
        for call in fetches
        if isinstance(call[1].downloader, pooch.HTTPDownloader)
    )
    assert [call[2].dataset for call in fetches] == ["monkey-vision", "nod-meg"]


def test_data_path_none_uses_os_cache_and_fetches_both(monkeypatch, tmp_path):
    fetched: list[str] = []

    class Fetcher:
        abspath = tmp_path

        def fetch(self, fname, **_kwargs):
            fetched.append(fname)

    monkeypatch.setattr(sample.pooch, "os_cache", lambda name: tmp_path if name == "vneurotk" else None)
    monkeypatch.setattr(sample.pooch, "create", lambda **_kwargs: Fetcher())

    assert sample.data_path(progressbar=False) == tmp_path / "vneurotk-samples"
    assert fetched == [sample._NOD_FNAME, sample._MV_FNAME]


def test_data_path_preserves_empty_explicit_path(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    seen: dict[str, object] = {}

    class Fetcher:
        abspath = tmp_path

        def fetch(self, *_args, **_kwargs):
            pass

    def fake_create(**kwargs):
        seen.update(kwargs)
        return Fetcher()

    monkeypatch.setattr(sample.pooch, "create", fake_create)
    monkeypatch.setattr(sample.pooch, "os_cache", lambda _name: pytest.fail("empty path is explicit"))
    sample.data_path("nod-meg", path="", progressbar=False)
    assert seen["path"] == Path("")


def _fetch_with_fake_download(tmp_path: Path, payload: bytes, *, expected_size: int, max_size: int) -> Path:
    cache = tmp_path / "cache"
    digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    fetcher = pooch.create(
        path=cache,
        base_url="https://invalid.example/",
        registry={"sample.zip": digest},
    )

    def fake_http(_url, output_file, _pooch):
        midpoint = max(1, len(payload) // 2)
        output_file.write(payload[:midpoint])
        output_file.write(payload[midpoint:])

    downloader = sample._BoundedHTTPDownloader(
        expected_size=expected_size,
        max_size=max_size,
        downloader=fake_http,
        progressbar=False,
    )
    return Path(fetcher.fetch("sample.zip", downloader=downloader))


def test_bounded_downloader_accepts_exact_write(tmp_path):
    payload = b"exact archive bytes"

    downloaded = _fetch_with_fake_download(
        tmp_path,
        payload,
        expected_size=len(payload),
        max_size=len(payload),
    )

    assert downloaded.read_bytes() == payload


def test_bounded_downloader_rejects_oversized_write_and_pooch_cleans_temp(tmp_path):
    payload = b"oversized archive bytes"

    with pytest.raises(ValueError, match="exceeds the .*byte limit"):
        _fetch_with_fake_download(
            tmp_path,
            payload,
            expected_size=len(payload) - 1,
            max_size=len(payload) - 1,
        )

    assert not (tmp_path / "cache/sample.zip").exists()
    assert list((tmp_path / "cache").iterdir()) == []


def test_bounded_downloader_rejects_truncated_write_and_pooch_cleans_temp(tmp_path):
    payload = b"truncated archive bytes"

    with pytest.raises(ValueError, match=rf"has {len(payload):,} bytes; expected {len(payload) + 1:,}"):
        _fetch_with_fake_download(
            tmp_path,
            payload,
            expected_size=len(payload) + 1,
            max_size=len(payload) + 1,
        )

    assert not (tmp_path / "cache/sample.zip").exists()
    assert list((tmp_path / "cache").iterdir()) == []


def test_real_pooch_fetch_verifies_and_extracts_in_memory_zip(tmp_path):
    source = tmp_path / "source.zip"
    digest = _write_zip(
        source,
        {
            "nod-meg/meg/recording.fif": b"recording",
            "nod-meg/stimuli/image.JPEG": b"pixels",
        },
    )
    cache = tmp_path / "cache"
    fetcher = pooch.create(
        path=cache,
        base_url="https://invalid.example/",
        registry={"sample.zip": digest},
        urls={"sample.zip": source.as_uri()},
    )

    def downloader(_url, output_file, _pooch):
        shutil.copyfile(source, output_file)

    result = fetcher.fetch(
        "sample.zip",
        downloader=downloader,
        processor=sample._SafeZipProcessor("nod-meg", digest),
    )

    extracted = Path(result)
    assert (extracted / "meg/recording.fif").read_bytes() == b"recording"
    assert (extracted / "stimuli/image.JPEG").read_bytes() == b"pixels"
    assert (cache / "vneurotk-samples/.nod-meg.complete").read_text() == f"{digest}\n"


def test_real_pooch_rejects_hash_mismatch_before_processing(tmp_path):
    source = tmp_path / "source.zip"
    _write_zip(source, {"nod-meg/file": b"data"})
    processor_called = False

    def processor(*_args):
        nonlocal processor_called
        processor_called = True

    fetcher = pooch.create(
        path=tmp_path / "cache",
        base_url="https://invalid.example/",
        registry={"sample.zip": "sha256:" + "0" * 64},
        urls={"sample.zip": source.as_uri()},
    )

    def downloader(_url, output_file, _pooch):
        shutil.copyfile(source, output_file)

    with pytest.raises(ValueError, match="SHA256 hash"):
        fetcher.fetch("sample.zip", downloader=downloader, processor=processor)
    assert not processor_called
    assert not (tmp_path / "cache/sample.zip").exists()


def test_digest_bound_marker_reuses_completed_extraction(tmp_path, monkeypatch):
    archive = tmp_path / "archive.zip"
    digest = _write_zip(archive, {"nod-meg/file.txt": b"original"})
    processor = sample._SafeZipProcessor("nod-meg", digest)
    first = Path(processor(str(archive), "download", None))

    monkeypatch.setattr(sample.zipfile, "ZipFile", lambda *_args, **_kwargs: pytest.fail("archive reopened"))
    second = Path(processor(str(archive), "fetch", None))

    assert second == first
    assert (second / "file.txt").read_bytes() == b"original"


def test_concurrent_thread_extraction_waits_and_reuses(tmp_path, monkeypatch):
    archive = tmp_path / "archive.zip"
    digest = _write_zip(archive, {"nod-meg/file.txt": b"thread payload"})
    processor = sample._SafeZipProcessor("nod-meg", digest)
    first_started = threading.Event()
    release_first = threading.Event()
    original = sample._extract_member
    calls = 0

    def pause_first(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            first_started.set()
            assert release_first.wait(timeout=2)
        return original(*args, **kwargs)

    monkeypatch.setattr(sample, "_extract_member", pause_first)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(processor, str(archive), "fetch", None)
        assert first_started.wait(timeout=2)
        second = pool.submit(processor, str(archive), "fetch", None)
        time.sleep(0.1)
        assert not second.done()
        release_first.set()
        results = [first.result(timeout=2), second.result(timeout=2)]

    destination = tmp_path / "vneurotk-samples/nod-meg"
    marker = tmp_path / "vneurotk-samples/.nod-meg.complete"
    assert results == [str(destination), str(destination)]
    assert calls == 1
    assert (destination / "file.txt").read_bytes() == b"thread payload"
    assert marker.read_text(encoding="ascii") == f"{digest}\n"
    assert not (tmp_path / "vneurotk-samples/.nod-meg.lock").exists()


def test_concurrent_process_extraction_both_succeed(tmp_path):
    archive = tmp_path / "archive.zip"
    digest = _write_zip(archive, {"nod-meg/file.txt": b"process payload" * 100_000})
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(target=_process_extract, args=(str(archive), digest, start, results)) for _ in range(2)
    ]
    for process in processes:
        process.start()
    start.set()
    outcomes = [results.get(timeout=10) for _ in processes]
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    destination = tmp_path / "vneurotk-samples/nod-meg"
    marker = tmp_path / "vneurotk-samples/.nod-meg.complete"
    assert outcomes == [(True, str(destination)), (True, str(destination))]
    assert (destination / "file.txt").read_bytes() == b"process payload" * 100_000
    assert marker.read_text(encoding="ascii") == f"{digest}\n"
    assert not (tmp_path / "vneurotk-samples/.nod-meg.lock").exists()


def test_stale_extraction_lock_is_recovered(tmp_path, monkeypatch):
    archive = tmp_path / "archive.zip"
    digest = _write_zip(archive, {"nod-meg/file.txt": b"recovered"})
    extract_root = tmp_path / "vneurotk-samples"
    extract_root.mkdir()
    lock = extract_root / ".nod-meg.lock"
    lock.write_text(
        json.dumps({"pid": 2**30, "hostname": socket.gethostname(), "created": 0, "token": "stale"}),
        encoding="ascii",
    )
    old = time.time() - 10
    os.utime(lock, (old, old))
    monkeypatch.setattr(sample, "_EXTRACTION_LOCK_STALE_AGE", 1.0)

    result = sample._SafeZipProcessor("nod-meg", digest)(str(archive), "fetch", None)

    destination = extract_root / "nod-meg"
    assert result == str(destination)
    assert (destination / "file.txt").read_bytes() == b"recovered"
    assert not lock.exists()


def test_extraction_lock_wait_is_bounded(tmp_path, monkeypatch):
    lock = tmp_path / ".nod-meg.lock"
    lock.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "created": time.time(),
                "token": "active",
            }
        ),
        encoding="ascii",
    )
    monkeypatch.setattr(sample, "_EXTRACTION_LOCK_TIMEOUT", 0.01)
    monkeypatch.setattr(sample, "_LOCK_INITIAL_BACKOFF", 0.001)
    monkeypatch.setattr(sample, "_LOCK_MAX_BACKOFF", 0.001)

    with pytest.raises(TimeoutError, match=r"Timed out after 0.01 seconds"):
        with sample._ExtractionLock(lock):
            pytest.fail("active lock must not be acquired")

    assert lock.exists()


def test_extraction_failure_releases_lock_and_marker_temp(tmp_path, monkeypatch):
    archive = tmp_path / "archive.zip"
    digest = _write_zip(archive, {"nod-meg/file.txt": b"payload"})

    def fail_marker(*args, **kwargs):
        raise OSError("marker temp failure")

    monkeypatch.setattr(sample.tempfile, "NamedTemporaryFile", fail_marker)
    with pytest.raises(OSError, match="marker temp failure"):
        sample._SafeZipProcessor("nod-meg", digest)(str(archive), "fetch", None)

    extract_root = tmp_path / "vneurotk-samples"
    assert not (extract_root / ".nod-meg.lock").exists()
    assert not list(extract_root.glob(".nod-meg.complete.*.tmp"))
    assert not list(extract_root.glob(".nod-meg-[!b]*"))


def test_wrong_digest_marker_forces_reextraction(tmp_path):
    archive = tmp_path / "archive.zip"
    digest = _write_zip(archive, {"nod-meg/file.txt": b"new"})
    root = tmp_path / "vneurotk-samples"
    destination = root / "nod-meg"
    destination.mkdir(parents=True)
    (destination / "file.txt").write_bytes(b"old")
    (root / ".nod-meg.complete").write_text("sha256:old\n")

    sample._SafeZipProcessor("nod-meg", digest)(str(archive), "fetch", None)

    assert (destination / "file.txt").read_bytes() == b"new"
    assert (root / ".nod-meg.complete").read_text() == f"{digest}\n"


def test_interrupted_extraction_preserves_existing_tree_and_marker(tmp_path, monkeypatch):
    archive = tmp_path / "archive.zip"
    old_digest = "sha256:old"
    new_digest = _write_zip(archive, {"nod-meg/one": b"one", "nod-meg/two": b"two"})
    root = tmp_path / "vneurotk-samples"
    destination = root / "nod-meg"
    destination.mkdir(parents=True)
    (destination / "old").write_bytes(b"old")
    marker = root / ".nod-meg.complete"
    marker.write_text(f"{old_digest}\n")
    original = sample._extract_member
    calls = 0

    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return original(*args, **kwargs)

    monkeypatch.setattr(sample, "_extract_member", interrupt)
    with pytest.raises(KeyboardInterrupt):
        sample._SafeZipProcessor("nod-meg", new_digest)(str(archive), "update", None)

    assert (destination / "old").read_bytes() == b"old"
    assert marker.read_text() == f"{old_digest}\n"
    assert not list(root.glob(".nod-meg-[!b]*"))


def test_interrupted_marker_install_restores_existing_tree(tmp_path, monkeypatch):
    archive = tmp_path / "archive.zip"
    old_digest = "sha256:old"
    new_digest = _write_zip(archive, {"nod-meg/new": b"new"})
    root = tmp_path / "vneurotk-samples"
    destination = root / "nod-meg"
    destination.mkdir(parents=True)
    (destination / "old").write_bytes(b"old")
    marker = root / ".nod-meg.complete"
    marker.write_text(f"{old_digest}\n")
    original_replace = sample.os.replace

    def interrupt_marker(source, target):
        if Path(target) == marker:
            raise KeyboardInterrupt
        original_replace(source, target)

    monkeypatch.setattr(sample.os, "replace", interrupt_marker)
    with pytest.raises(KeyboardInterrupt):
        sample._SafeZipProcessor("nod-meg", new_digest)(str(archive), "update", None)

    assert (destination / "old").read_bytes() == b"old"
    assert not (destination / "new").exists()
    assert marker.read_text() == f"{old_digest}\n"
    assert not (root / ".nod-meg.lock").exists()
    assert not list(root.glob(".nod-meg.complete.*.tmp"))


@pytest.mark.parametrize(
    "member",
    [
        "/nod-meg/file",
        "../nod-meg/file",
        "nod-meg/../escape",
        "C:/nod-meg/file",
        r"C:\\nod-meg\\file",
        r"nod-meg\\..\\escape",
        "monkey-vision/file",
    ],
)
def test_safe_processor_rejects_unsafe_or_cross_dataset_paths(tmp_path, member):
    archive = tmp_path / "bad.zip"
    digest = _write_zip(archive, {member: b"bad"})

    with pytest.raises(ValueError, match="Unsafe|outside"):
        sample._SafeZipProcessor("nod-meg", digest)(str(archive), "download", None)
    assert not (tmp_path / "vneurotk-samples/nod-meg").exists()


def test_safe_processor_rejects_symlink(tmp_path):
    archive = tmp_path / "symlink.zip"
    with zipfile.ZipFile(archive, "w") as output:
        info = zipfile.ZipInfo("nod-meg/link")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        output.writestr(info, "target")
    digest = f"sha256:{hashlib.sha256(archive.read_bytes()).hexdigest()}"

    with pytest.raises(ValueError, match="symbolic link"):
        sample._SafeZipProcessor("nod-meg", digest)(str(archive), "download", None)


def test_safe_processor_never_uses_extractall(tmp_path, monkeypatch):
    archive = tmp_path / "archive.zip"
    digest = _write_zip(archive, {"nod-meg/file": b"ok"})
    monkeypatch.setattr(zipfile.ZipFile, "extractall", lambda *_args, **_kwargs: pytest.fail("unsafe extractall"))
    sample._SafeZipProcessor("nod-meg", digest)(str(archive), "download", None)


def test_member_count_limit(tmp_path, monkeypatch):
    archive = tmp_path / "archive.zip"
    digest = _write_zip(archive, {"nod-meg/one": b"1", "nod-meg/two": b"2"})
    monkeypatch.setattr(sample, "_MAX_ARCHIVE_MEMBERS", 1)
    with pytest.raises(ValueError, match="more than 1 members"):
        sample._SafeZipProcessor("nod-meg", digest)(str(archive), "download", None)


def test_individual_size_limit(tmp_path, monkeypatch):
    archive = tmp_path / "archive.zip"
    digest = _write_zip(archive, {"nod-meg/file": b"12"})
    monkeypatch.setattr(sample, "_MAX_MEMBER_SIZE", 1)
    with pytest.raises(ValueError, match="1-byte limit"):
        sample._SafeZipProcessor("nod-meg", digest)(str(archive), "download", None)


def test_total_size_limit(tmp_path, monkeypatch):
    archive = tmp_path / "archive.zip"
    digest = _write_zip(archive, {"nod-meg/one": b"1", "nod-meg/two": b"2"})
    monkeypatch.setattr(sample, "_MAX_TOTAL_SIZE", 1)
    with pytest.raises(ValueError, match="1-byte uncompressed limit"):
        sample._SafeZipProcessor("nod-meg", digest)(str(archive), "download", None)


def test_compression_ratio_limit(tmp_path, monkeypatch):
    archive = tmp_path / "archive.zip"
    digest = _write_zip(archive, {"nod-meg/file": b"0" * 10_000})
    monkeypatch.setattr(sample, "_MAX_COMPRESSION_RATIO", 2)
    with pytest.raises(ValueError, match="2:1 compression-ratio limit"):
        sample._SafeZipProcessor("nod-meg", digest)(str(archive), "download", None)


def test_authoritative_archive_constants():
    assert sample._NOD_HASH == "sha256:cebcec0bab57d548c486d7e4456c1e56c5832a47e4793ddb1e8f5f6a4d403968"
    assert sample._NOD_SIZE == 87_243_257
    assert sample._NOD_MAX_SIZE == sample._NOD_SIZE
    assert sample._MV_HASH == "sha256:bb5c1a8aab1faa4fb97d89adaabfa301bea7f4879b54b102f3b3a17f35ada94e"
    assert sample._MV_SIZE == 191_862_733
    assert sample._MV_MAX_SIZE == sample._MV_SIZE
    assert sample._MAX_ARCHIVE_MEMBERS == 10_000
    assert sample._MAX_MEMBER_SIZE == 2 * 1024**3
    assert sample._MAX_TOTAL_SIZE == 10 * 1024**3
    assert sample._MAX_COMPRESSION_RATIO == 1_000
