"""Neural-domain primitives for VneuroTK.

Provides :class:`NeuroData`: the neural signal container with trial-structured
views.  Trial structure computation has moved to :mod:`vneurotk.neuro.trial`.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

_STYLE = (
    "<style scoped>"
    ".nd-info{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',"
    "Roboto,sans-serif;font-size:13px;max-width:420px}"
    ".nd-info table{width:100%;border-collapse:collapse;margin:4px 0}"
    ".nd-info th,.nd-info td{padding:3px 10px;border-bottom:1px solid currentColor;"
    "border-bottom-opacity:0.2}"
    ".nd-info th{text-align:left;width:50%;font-weight:500;opacity:0.75}"
    ".nd-info td{text-align:right}"
    ".nd-info tr:last-child th,.nd-info tr:last-child td{border-bottom:none}"
    ".nd-info .nd-ok{color:#2a9d2a}.nd-info .nd-na{opacity:0.5;font-style:italic}"
    "</style>"
)


class NeuroData:
    """Neural signal container with trial-structured views.

    Holds the raw neural array plus optional trial-boundary information, and
    exposes ``epochs`` and ``continuous`` views derived from that structure.
    ``NeuroData`` is *not* a NumPy array subclass — use :attr:`data` for the
    raw array when NumPy operations are needed.

    Parameters
    ----------
    data : np.ndarray
        Raw neural array.
    trial_starts : np.ndarray or None
        Start sample index per trial.
    trial_ends : np.ndarray or None
        End sample index per trial.
    data_mode : str or None
        ``"continuous"``, ``"epochs"``, or ``"patterns"``.

    Examples
    --------
    >>> import numpy as np
    >>> nd = NeuroData(np.random.randn(1000, 64))
    >>> nd.shape
    (1000, 64)
    >>> nd.data[:100]   # plain ndarray slice
    """

    def __init__(
        self,
        data: np.ndarray,
        trial_starts: np.ndarray | None = None,
        trial_ends: np.ndarray | None = None,
        data_mode: str | None = None,
    ) -> None:
        self._data = np.asarray(data)
        self._trial_starts = trial_starts
        self._trial_ends = trial_ends
        self._data_mode = data_mode

    # ------------------------------------------------------------------
    # Raw data access
    # ------------------------------------------------------------------

    @property
    def data(self) -> np.ndarray:
        """Raw neural array as a plain :class:`numpy.ndarray`."""
        return self._data

    # ------------------------------------------------------------------
    # ndarray-compatible attribute proxies
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple:
        """Shape of the underlying data array."""
        return self._data.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions of the underlying data array."""
        return self._data.ndim

    @property
    def dtype(self) -> np.dtype:
        """Data type of the underlying data array."""
        return self._data.dtype

    @property
    def size(self) -> int:
        """Total number of elements."""
        return self._data.size

    def __array__(self, dtype=None) -> np.ndarray:
        """Support ``np.asarray(neuro_data)``."""
        if dtype is not None:
            return self._data.astype(dtype)
        return self._data

    # ------------------------------------------------------------------
    # Trial-structured views
    # ------------------------------------------------------------------

    @property
    def epochs(self) -> np.ndarray:
        """Trial-epoched view, shape ``(n_trials, n_timepoints, nchan)``.

        If the underlying data is already in epochs format it is returned
        as-is with a warning.

        Raises
        ------
        RuntimeError
            If trial structure is not available (call ``BaseData.configure()`` first).
        """
        if self._trial_starts is None:
            raise RuntimeError("No trial structure. Call BaseData.configure() first.")

        if self._data_mode == "epochs":
            logger.warning(
                "neuro is already in epochs format (shape {}), returning as-is",
                self._data.shape,
            )
            return self._data

        return np.stack([self._data[s:e] for s, e in zip(self._trial_starts, self._trial_ends, strict=True)])  # ty: ignore[not-iterable, invalid-argument-type]

    @property
    def continuous(self) -> np.ndarray:
        """Concatenated-trials view, shape ``(total_trial_samples, nchan)``.

        If the underlying data is already in continuous format it is returned
        as-is with a warning.

        Raises
        ------
        RuntimeError
            If trial structure is not available (call ``BaseData.configure()`` first).
        """
        if self._trial_starts is None:
            raise RuntimeError("No trial structure. Call BaseData.configure() first.")

        if self._data_mode == "continuous":
            logger.warning(
                "neuro is already in continuous format (shape {}), returning as-is",
                self._data.shape,
            )
            return self._data

        if self._data_mode == "epochs":
            return self._data.reshape(-1, self._data.shape[-1])

        return np.concatenate([self._data[s:e] for s, e in zip(self._trial_starts, self._trial_ends, strict=True)])  # ty: ignore[not-iterable, invalid-argument-type]

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _has_trial_structure(self) -> bool:
        return self._trial_starts is not None

    def __repr__(self) -> str:
        mode = self._data_mode or "unknown"
        ts = "configured" if self._has_trial_structure() else "not configured"
        n_trials = len(self._trial_starts) if self._trial_starts is not None else 0
        return f"NeuroData  shape={self._data.shape}  mode={mode}" + (
            f"  trials={n_trials}" if n_trials else f"  trials={ts}"
        )

    def _repr_html_(self) -> str:
        mode = self._data_mode or "unknown"
        has_ts = self._has_trial_structure()
        n_trials = len(self._trial_starts) if self._trial_starts is not None else None

        ok = '<span class="nd-ok">✓ available</span>'
        na = '<span class="nd-na">not configured</span>'

        rows = [
            ("Shape", str(self._data.shape)),
            ("Mode", mode),
            ("Trials", str(n_trials) if n_trials is not None else "—"),
            ("Channels", str(self._data.shape[-1]) if self._data.ndim >= 2 else "—"),
            (".epochs", ok if has_ts else na),
            (".continuous", ok if has_ts else na),
        ]
        trs = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
        table = f"<table>{trs}</table>"
        return (
            f'{_STYLE}<div class="nd-info">'
            f"<details open><summary><strong>NeuroData</strong></summary>{table}</details>"
            f"</div>"
        )
