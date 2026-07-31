"""Tests for opt-in, host-safe logging configuration."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from loguru import logger

import vneurotk as vtk

ROOT = Path(__file__).resolve().parents[1]


def _run_python(code: str) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout.splitlines()[-1])


def test_top_level_import_preserves_host_loguru_sinks() -> None:
    """A host sink keeps receiving records after the top-level import."""
    result = _run_python(
        """
import json
from loguru import logger
messages = []
host_id = logger.add(lambda message: messages.append(message.record["message"]), format="{message}")
logger.info("before")
import vneurotk
logger.info("after")
logger.remove(host_id)
print(json.dumps({"messages": messages}))
"""
    )
    assert result == {"messages": ["before", "after"]}


def test_top_level_import_does_not_touch_mne_or_warning_state() -> None:
    """Import neither loads MNE nor adds VneuroTK warning filters."""
    result = _run_python(
        """
import json
import sys
import warnings
import numpy  # establish NumPy's own process-global warning filters
before = list(warnings.filters)
import vneurotk
print(json.dumps({
    "mne_loaded": "mne" in sys.modules,
    "warnings_unchanged": warnings.filters == before,
}))
"""
    )
    assert result == {"mne_loaded": False, "warnings_unchanged": True}


def test_setup_logging_is_idempotent_and_preserves_host_sink() -> None:
    """Repeated setup replaces only VneuroTK's sink and uses the latest level."""
    host_messages: list[str] = []
    first_messages: list[str] = []
    second_messages: list[str] = []
    host_id = logger.add(lambda message: host_messages.append(message.record["message"]), format="{message}")
    try:
        vtk.setup_logging("INFO", sink=lambda message: first_messages.append(message.record["message"]), colorize=False)
        logger.info("host-before")
        logger.patch(lambda record: record.update(name="vneurotk.test")).info("package-first")

        vtk.setup_logging(
            "ERROR", sink=lambda message: second_messages.append(message.record["message"]), colorize=False
        )
        logger.info("host-after")
        package_logger = logger.patch(lambda record: record.update(name="vneurotk.test"))
        package_logger.info("package-info")
        package_logger.error("package-error")
    finally:
        logger.remove(host_id)

    assert first_messages == ["package-first"]
    assert second_messages == ["package-error"]
    assert host_messages == ["host-before", "package-first", "host-after", "package-info", "package-error"]


def test_set_log_level_only_configures_vneurotk_sink(monkeypatch) -> None:
    """The convenience API reads the environment without requesting MNE integration."""
    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setenv("VNTK_LOGGING_LEVEL", "WARNING")
    monkeypatch.setattr("vneurotk._log.setup_logging", lambda *args, **kwargs: calls.append((args, kwargs)))

    vtk.set_log_level()

    assert calls == [(("WARNING",), {})]


def test_mne_integration_is_explicit(monkeypatch) -> None:
    """Default setup leaves MNE and warnings alone; opt-in applies each setting once."""
    import warnings

    mne_calls: list[str] = []
    warning_calls: list[tuple[tuple, dict]] = []

    class FakeMNE:
        @staticmethod
        def set_log_level(level: str) -> None:
            mne_calls.append(level)

    monkeypatch.setitem(sys.modules, "mne", FakeMNE())
    monkeypatch.setattr(warnings, "filterwarnings", lambda *args, **kwargs: warning_calls.append((args, kwargs)))

    vtk.setup_logging("INFO", sink=lambda _: None, colorize=False)
    assert mne_calls == []
    assert warning_calls == []

    vtk.setup_logging(
        "INFO",
        sink=lambda _: None,
        colorize=False,
        mne_level="warning",
        suppress_mne_naming_warnings=True,
    )
    assert mne_calls == ["WARNING"]
    assert len(warning_calls) == 1
    assert warning_calls[0][0] == ("ignore",)
    assert warning_calls[0][1]["category"] is RuntimeWarning
