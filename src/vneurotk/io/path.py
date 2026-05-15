"""Path classes for VneuroTK data sources."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from vneurotk.core.recording import BaseData

EPHYS_DTYPES: frozenset[str] = frozenset(
    [
        "TrialRaster",
        "TrialRecord",
        "UnitProp",
        "MeanFr",
        "AvgPsth",
        "ChTrialRaster",
        "ChTrialRecord",
        "ChMeanFr",
        "ChProp",
        "ChStimFr",
    ]
)

EPHYS_EXTENSIONS: frozenset[str] = frozenset(["h5", "csv", "nwb"])


@dataclass
class VTKPath:
    """Base path class for VneuroTK data sources.

    Attributes
    ----------
    root : Path
        Root directory for the data.
    session : str | None
        Session identifier.
    subject : str | None
        Subject identifier.
    task : str | None
        Task identifier.
    run : str | None
        Run identifier.
    desc : str | None
        Description identifier.
    probe : int | None
        Probe number (ephys only).
    suffix : str | None
        File suffix.
    extension : str | None
        File extension.
    modality : str | None
        Data modality (ephys, mne, bids).
    """

    root: Path
    session: str | None = None
    subject: str | None = None
    task: str | None = None
    run: str | None = None
    desc: str | None = None
    probe: int | None = None
    suffix: str | None = None
    extension: str | None = None
    modality: str | None = None

    def __post_init__(self) -> None:
        """Convert root to Path if needed."""
        if not isinstance(self.root, Path):
            self.root = Path(self.root)

    @property
    def fpath(self) -> Path:
        """Construct full file path.

        Returns
        -------
        Path
            Full file path constructed from attributes.
        """
        # Build filename from non-None attributes
        parts = []
        if self.subject is not None:
            parts.append(f"sub-{self.subject}")
        if self.session is not None:
            parts.append(f"ses-{self.session}")
        if self.task is not None:
            parts.append(f"task-{self.task}")
        if self.run is not None:
            parts.append(f"run-{self.run}")
        if self.desc is not None:
            parts.append(f"desc-{self.desc}")
        if self.probe is not None:
            parts.append(f"probe-{self.probe}")
        if self.suffix is not None:
            parts.append(self.suffix)

        filename = "_".join(parts) if parts else "data"

        # Add extension
        ext = self.extension if self.extension else ".h5"
        if not ext.startswith("."):
            ext = f".{ext}"

        return self.root / f"{filename}{ext}"

    def load(self, pre_load: bool = False) -> BaseData:
        """Load data described by this path into a :class:`BaseData`.

        For base :class:`VTKPath` instances pointing to a saved ``.h5`` file,
        loads via the internal HDF5 reader.  Typed subclasses override this
        method with format-specific loading logic.

        Raises
        ------
        NotImplementedError
            If this path does not point to a ``.h5`` file and has no typed
            loading strategy (i.e., it is a plain :class:`VTKPath`).
        """
        fpath = self.fpath
        if fpath.suffix == ".h5":
            from vneurotk.io.loader import _load_from_h5

            bd = _load_from_h5(fpath)
            return bd.load() if pre_load else bd
        raise NotImplementedError(
            f"{type(self).__name__} does not implement load() for non-HDF5 paths. "
            "Use a typed subclass (EphysPath, MNEPath, BIDSPath)."
        )


@dataclass
class EphysPath(VTKPath):
    """Path class for ephys session-level data.

    Follows the naming convention::

        {root}/sessions/{session_id}/{dtype}_{session_id}[_probe{N}].{ext}

    The ``fpath`` property points to a file under ``sessions/``.
    Auxiliary properties ``raw_dir`` and ``nwb_path`` expose the ``raw/``
    and ``nwb/`` subdirectories for convenience.

    VTKPath fields ``subject``, ``task``, ``run``, ``desc``, and ``suffix``
    are inherited but not used by ``fpath``; use ``session_id`` and ``dtype``
    instead.

    Parameters
    ----------
    root : Path
        Root project directory (e.g. ``DB/ephys/MonkeyVision``).
    session_id : str, optional
        Full session identifier, e.g. ``"251024_FanFan_nsd1w_MSB"``.
        Format: ``{date}_{subject}_{paradigm}_{region}``.
    dtype : str, optional
        Data type.  Must be one of :data:`EPHYS_DTYPES`.
    probe : int, optional
        Probe index (0-based).  ``None`` for single-probe sessions.
    extension : str, optional
        File extension.  Must be one of :data:`EPHYS_EXTENSIONS`.
        Default is ``"h5"``.
    """

    session_id: str | None = None
    dtype: str | None = None
    modality: str = field(default="ephys", init=False)

    def __post_init__(self) -> None:
        """Validate dtype and extension."""
        super().__post_init__()
        if self.dtype is not None and self.dtype not in EPHYS_DTYPES:
            raise ValueError(f"Invalid dtype '{self.dtype}'. Must be one of {sorted(EPHYS_DTYPES)}")
        if self.extension is not None:
            ext = self.extension.lstrip(".")
            if ext not in EPHYS_EXTENSIONS:
                raise ValueError(f"Invalid extension '{ext}'. Must be one of {sorted(EPHYS_EXTENSIONS)}")

    # ------------------------------------------------------------------
    # Primary path: sessions/
    # ------------------------------------------------------------------

    @property
    def session_dir(self) -> Path:
        """Directory containing this session's analysis files.

        Returns
        -------
        Path
            ``{root}/sessions/{session_id}``

        Raises
        ------
        ValueError
            If ``session_id`` is not set.
        """
        if self.session_id is None:
            raise ValueError("session_id must be set to access session_dir")
        return self.root / "sessions" / self.session_id

    @property
    def fpath(self) -> Path:
        """Full path to the session-level analysis file.

        Returns
        -------
        Path
            ``{root}/sessions/{session_id}/{dtype}_{session_id}[_probe{N}].{ext}``

        Raises
        ------
        ValueError
            If ``session_id`` or ``dtype`` is not set.
        """
        if self.session_id is None:
            raise ValueError("session_id must be set to construct fpath")
        if self.dtype is None:
            raise ValueError("dtype must be set to construct fpath")
        probe_tag = f"_probe{self.probe}" if self.probe is not None else ""
        filename = f"{self.dtype}_{self.session_id}{probe_tag}"
        ext = (self.extension or "h5").lstrip(".")
        return self.session_dir / f"{filename}.{ext}"

    # ------------------------------------------------------------------
    # Auxiliary paths: raw/ and nwb/
    # ------------------------------------------------------------------

    @property
    def raw_dir(self) -> Path:
        """Raw data directory for this session.

        Returns
        -------
        Path
            ``{root}/raw/{session_id}``

        Raises
        ------
        ValueError
            If ``session_id`` is not set.
        """
        if self.session_id is None:
            raise ValueError("session_id must be set to access raw_dir")
        return self.root / "raw" / self.session_id

    @property
    def nwb_path(self) -> Path:
        """Path to the NWB intermediate file for this session.

        Returns
        -------
        Path
            ``{root}/nwb/{session_id}[_probe{N}].nwb``

        Raises
        ------
        ValueError
            If ``session_id`` is not set.
        """
        if self.session_id is None:
            raise ValueError("session_id must be set to access nwb_path")
        probe_tag = f"_probe{self.probe}" if self.probe is not None else ""
        return self.root / "nwb" / f"{self.session_id}{probe_tag}.nwb"

    # ------------------------------------------------------------------
    # Convenience constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_components(
        cls,
        root: str | Path,
        date: str,
        subject: str,
        paradigm: str,
        region: str,
        dtype: str | None = None,
        probe: int | None = None,
        extension: str = "h5",
    ) -> EphysPath:
        """Build an EphysPath from individual session components.

        Parameters
        ----------
        root : str or Path
            Root project directory.
        date : str
            Session date, e.g. ``"251024"``.
        subject : str
            Subject name, e.g. ``"FanFan"``.
        paradigm : str
            Paradigm string including optional block/run marker,
            e.g. ``"nsd1w"`` (paradigm ``nsd``, block ``1w``).
        region : str
            Brain region, e.g. ``"MSB"``.
        dtype : str, optional
            Data type (see :data:`EPHYS_DTYPES`).
        probe : int, optional
            Probe index (0-based).
        extension : str
            File extension.  Default ``"h5"``.

        Returns
        -------
        EphysPath
        """
        session_id = f"{date}_{subject}_{paradigm}_{region}"
        return cls(
            root=Path(root),
            session_id=session_id,
            dtype=dtype,
            probe=probe,
            extension=extension,
        )

    def load(self, pre_load: bool = False) -> BaseData:
        """Load this Ephys session into a :class:`~vneurotk.neuro.base.BaseData`.

        Parameters
        ----------
        pre_load : bool
            If ``True``, eagerly load neuro data into memory before returning.

        Returns
        -------
        BaseData
        """
        from vneurotk.io.loader import _load_from_ephys

        bd = _load_from_ephys(self)
        return bd.load() if pre_load else bd


@dataclass
class MNEPath(VTKPath):
    """Path class for MNE data.

    Inherits all attributes from VTKPath.
    Sets modality to 'mne' by default.
    Constructs MNE-style file paths.
    """

    modality: str = field(default="mne", init=False)

    @property
    def fpath(self) -> Path:
        """Construct MNE-style file path.

        Returns
        -------
        Path
            Full file path in MNE format:
            root/sub-{subject}_ses-{session}_task-{task}_run-{run}_{suffix}.{extension}
        """
        parts = []
        if self.subject is not None:
            parts.append(f"sub-{self.subject}")
        if self.session is not None:
            parts.append(f"ses-{self.session}")
        if self.task is not None:
            parts.append(f"task-{self.task}")
        if self.run is not None:
            parts.append(f"run-{self.run}")

        filename = "_".join(parts) if parts else "data"

        if self.suffix is not None:
            filename = f"{filename}_{self.suffix}"

        ext = self.extension if self.extension else ".fif"
        if not ext.startswith("."):
            ext = f".{ext}"

        return self.root / f"{filename}{ext}"

    def load(self, pre_load: bool = False) -> BaseData:
        """Load this MNE recording into a :class:`~vneurotk.neuro.base.BaseData`.

        Parameters
        ----------
        pre_load : bool
            If ``True``, eagerly load neuro data into memory before returning.

        Returns
        -------
        BaseData
        """
        from vneurotk.io.loader import _load_from_mne

        bd = _load_from_mne(self)
        return bd.load() if pre_load else bd


@dataclass
class BIDSPath(VTKPath):
    """Path class for BIDS data.

    Inherits all attributes from VTKPath.
    Sets modality to 'bids' by default.
    Wraps mne_bids.BIDSPath internally.
    """

    modality: str = field(default="bids", init=False)
    _bids_path: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize BIDS path wrapper."""
        super().__post_init__()
        try:
            from mne_bids import BIDSPath as MNEBIDSPath  # type: ignore

            self._bids_path = MNEBIDSPath(
                root=self.root,
                subject=self.subject,
                session=self.session,
                task=self.task,
                run=self.run,
                suffix=self.suffix,
                extension=self.extension,
            )
        except ImportError:
            logger.warning("mne_bids not available, BIDSPath functionality limited")
            self._bids_path = None

    @property
    def fpath(self) -> Path:
        """Get BIDS file path.

        Returns
        -------
        Path
            Full BIDS file path.
        """
        if self._bids_path is not None:
            return Path(self._bids_path.fpath)
        else:
            # Fallback to basic path construction
            return super().fpath

    @property
    def bids_path(self) -> Any:
        """Get underlying mne_bids.BIDSPath object.

        Returns
        -------
        mne_bids.BIDSPath
            The wrapped BIDS path object.
        """
        return self._bids_path

    def load(self, pre_load: bool = False) -> BaseData:
        """Load this BIDS recording into a :class:`~vneurotk.neuro.base.BaseData`.

        Parameters
        ----------
        pre_load : bool
            If ``True``, eagerly load neuro data into memory before returning.

        Returns
        -------
        BaseData
        """
        from vneurotk.io.loader import _load_from_bids

        bd = _load_from_bids(self)
        return bd.load() if pre_load else bd
