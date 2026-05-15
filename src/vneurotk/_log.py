"""Logging configuration for vneurotk.

By default vneurotk logs at INFO and MNE at ERROR.
Call :func:`set_log_level` or :func:`setup_logging` to change levels.

Examples
--------
>>> import vneurotk as vtk
>>> vtk.set_log_level("DEBUG")     # explicit level
>>> vtk.set_log_level()            # reads VNTK_LOGGING_LEVEL, defaults to INFO
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from loguru import Record

__all__ = ["setup_logging", "set_log_level"]

_FMT_DEBUG = (
    "<green>{time:MMDD-HH:mm}</green>|"
    "<level>{level: ^4}</level>|"
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)
_FMT_INFO = "<level>{message}</level>"
_FMT_WARNING = "<level>{level: ^4}</level> | <level>{message}</level>"
_FMT_ERROR = (
    "<level>{level: ^4}</level>|"
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

_SINK_ID: int | None = None


def _formatter(record: Record) -> str:
    no = record["level"].no
    if no < 20:  # DEBUG / TRACE
        fmt = _FMT_DEBUG
    elif no < 30:  # INFO
        fmt = _FMT_INFO
    elif no < 40:  # WARNING
        fmt = _FMT_WARNING
    else:  # ERROR / CRITICAL
        fmt = _FMT_ERROR
    return fmt + "\n{exception}"


def setup_logging(
    level: str = "INFO",
    sink=None,
    *,
    colorize: bool = True,
    mne_level: str = "ERROR",
) -> None:
    """Configure vneurotk logging.

    Parameters
    ----------
    level : str
        Minimum log level for vneurotk: ``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
        ``"ERROR"``, or ``"CRITICAL"``.  Default ``"INFO"``.
    sink : file-like or None
        Output sink passed to ``loguru.logger.add()``.
        Defaults to ``sys.stderr``.
    colorize : bool
        Enable ANSI colour codes.  Default ``True``.
    mne_level : str
        Minimum log level for MNE-Python.  Default ``"ERROR"``.
    """
    global _SINK_ID

    if sink is None:
        sink = sys.stderr

    # remove loguru's default stderr sink and any previous vtk sink
    try:
        logger.remove(0)
    except ValueError:
        pass
    if _SINK_ID is not None:
        try:
            logger.remove(_SINK_ID)
        except ValueError:
            pass

    logger.enable("vneurotk")
    _SINK_ID = logger.add(
        sink,
        format=_formatter,
        level=level.upper(),
        colorize=colorize,
        filter="vneurotk",
    )

    try:
        import mne  # type: ignore

        mne.set_log_level(mne_level.upper())
        warnings.filterwarnings(
            "ignore",
            message=r".*does not conform to MNE naming conventions.*",
            category=RuntimeWarning,
        )
    except ImportError:
        pass


def set_log_level(verbose: str | None = None) -> None:
    """Set the vneurotk log level.

    Parameters
    ----------
    verbose : str or None
        Log level: ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``,
        or ``"CRITICAL"``.  If ``None``, reads the ``VNTK_LOGGING_LEVEL``
        environment variable; falls back to ``"INFO"`` if unset.

    Examples
    --------
    >>> vtk.set_log_level("DEBUG")   # explicit
    >>> vtk.set_log_level()          # from env or default INFO
    """
    level = verbose or os.environ.get("VNTK_LOGGING_LEVEL", "INFO")
    setup_logging(level)
