"""Opt-in logging configuration for VneuroTK.

Importing :mod:`vneurotk` does not configure Loguru, alter host sinks, or change
MNE-Python logging and warning policy. Applications can call
:func:`set_log_level` or :func:`setup_logging` when they want a
VneuroTK-owned, package-filtered sink.

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
from typing import TYPE_CHECKING, Any

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
    sink: Any = None,
    *,
    colorize: bool = True,
    mne_level: str | None = None,
    suppress_mne_naming_warnings: bool = False,
) -> None:
    """Configure an idempotent, VneuroTK-owned logging sink.

    The sink is filtered to records from the ``vneurotk`` package. Existing
    Loguru sinks belong to the host application and are never removed or
    reconfigured. Repeated calls replace only the sink previously created by
    this function.

    MNE integration is separately opt-in because both MNE's log level and
    Python warning filters are process-global. Pass ``mne_level`` and/or
    ``suppress_mne_naming_warnings=True`` only when the application wants
    VneuroTK to own those settings.

    Parameters
    ----------
    level : str
        Minimum level for the VneuroTK-owned sink: ``"DEBUG"``, ``"INFO"``,
        ``"WARNING"``, ``"ERROR"``, or ``"CRITICAL"``. Default ``"INFO"``.
    sink : file-like or None
        Output sink passed to :meth:`loguru.Logger.add`. Defaults to
        ``sys.stderr``.
    colorize : bool
        Enable ANSI colour codes on the VneuroTK-owned sink. Default ``True``.
    mne_level : str or None
        When provided, explicitly set MNE-Python's process-global log level.
        ``None`` (the default) leaves MNE unchanged.
    suppress_mne_naming_warnings : bool
        Explicitly install a process-global warning filter for MNE channel
        naming-convention warnings. Default ``False`` leaves warning filters
        unchanged.
    """
    global _SINK_ID

    if sink is None:
        sink = sys.stderr

    if _SINK_ID is not None:
        try:
            logger.remove(_SINK_ID)
        except ValueError:
            # A host may have removed all sinks since our previous call.
            pass

    logger.enable("vneurotk")
    _SINK_ID = logger.add(
        sink,
        format=_formatter,
        level=level.upper(),
        colorize=colorize,
        filter="vneurotk",
    )

    if mne_level is not None:
        try:
            import mne  # type: ignore
        except ImportError:
            pass
        else:
            mne.set_log_level(mne_level.upper())

    if suppress_mne_naming_warnings:
        warnings.filterwarnings(
            "ignore",
            message=r".*does not conform to MNE naming conventions.*",
            category=RuntimeWarning,
        )


def set_log_level(verbose: str | None = None) -> None:
    """Set the level of the VneuroTK-owned sink.

    This function does not alter host Loguru sinks or MNE settings. Loguru
    sinks configured by the host may independently choose to receive
    VneuroTK records.

    Parameters
    ----------
    verbose : str or None
        Log level: ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``,
        or ``"CRITICAL"``. If ``None``, read ``VNTK_LOGGING_LEVEL`` and fall
        back to ``"INFO"`` when it is unset.

    Examples
    --------
    >>> import vneurotk as vtk
    >>> vtk.set_log_level("DEBUG")
    >>> vtk.set_log_level()  # from VNTK_LOGGING_LEVEL or INFO
    """
    level = verbose or os.environ.get("VNTK_LOGGING_LEVEL", "INFO")
    setup_logging(level)
