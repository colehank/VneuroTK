"""Local model cache discovery and display."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from rich.console import Console

__all__ = ["CachedModel", "find_cached_models", "print_cached_models"]

Source = Literal["transformers", "timm", "torch", "clip", "keras"]

_RICH_COLORS: dict[str, str] = {
    "transformers": "bright_white",
    "timm": "cyan",
    "torch": "yellow",
    "clip": "magenta",
    "keras": "green",
}


@dataclass
class CachedModel:
    """Metadata for a locally cached model.

    Parameters
    ----------
    model_id : str
        Model identifier or filename.
    source : Source
        Cache origin: ``"transformers"``, ``"timm"``, ``"torch"``,
        ``"clip"``, or ``"keras"``.
    size_bytes : int
        Total size on disk in bytes.
    last_used : datetime
        Last access time.
    """

    model_id: str
    source: Source
    size_bytes: int
    last_used: datetime


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def _scan_hf_hub(cache_dir: Path) -> list[CachedModel]:
    """Scan HuggingFace hub cache for transformers and timm models."""
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return []

    try:
        info = scan_cache_dir(cache_dir=str(cache_dir))
    except Exception:
        return []

    results = []
    for repo in info.repos:
        if repo.repo_type != "model":
            continue
        source: Source = "timm" if repo.repo_id.startswith("timm/") else "transformers"
        results.append(
            CachedModel(
                model_id=repo.repo_id,
                source=source,
                size_bytes=repo.size_on_disk,
                last_used=datetime.fromtimestamp(repo.last_accessed),
            )
        )
    return results


def _scan_files(directory: Path, source: Source, suffixes: set[str]) -> list[CachedModel]:
    """Scan a flat directory for model files by suffix."""
    if not directory.is_dir():
        return []
    results = []
    for f in directory.iterdir():
        if f.suffix.lower() not in suffixes:
            continue
        stat = f.stat()
        results.append(
            CachedModel(
                model_id=f.name,
                source=source,
                size_bytes=stat.st_size,
                last_used=datetime.fromtimestamp(stat.st_atime),
            )
        )
    return results


def _scan_torch_hub(root: Path) -> list[CachedModel]:
    """Scan torch hub cache (subdirs + checkpoints/)."""
    if not root.is_dir():
        return []
    results = []
    # checkpoints subdir: flat .pth/.pt files
    results.extend(_scan_files(root / "checkpoints", "torch", {".pth", ".pt", ".bin"}))
    # top-level hub subdirs (e.g. pytorch_vision_v0.6.0)
    for d in root.iterdir():
        if d.is_dir() and d.name != "checkpoints":
            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            stat = d.stat()
            results.append(
                CachedModel(
                    model_id=d.name,
                    source="torch",
                    size_bytes=size,
                    last_used=datetime.fromtimestamp(stat.st_atime),
                )
            )
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_cached_models(hf_cache_dir: str | Path | None = None) -> list[CachedModel]:
    """Return all locally cached models across known cache locations.

    Parameters
    ----------
    hf_cache_dir : str or Path or None
        HuggingFace hub cache directory.  Defaults to
        ``huggingface_hub.constants.HF_HUB_CACHE``, which respects
        ``HF_HUB_CACHE`` and ``HF_HOME`` environment variables.

    Returns
    -------
    list[CachedModel]
        One entry per cached model, sorted by source then model_id.
    """
    home = Path.home()

    if hf_cache_dir:
        hf_dir = Path(hf_cache_dir)
    else:
        try:
            from huggingface_hub.constants import HF_HUB_CACHE

            hf_dir = Path(HF_HUB_CACHE)
        except ImportError:
            hf_dir = home / ".cache" / "huggingface" / "hub"

    results: list[CachedModel] = []
    results.extend(_scan_hf_hub(hf_dir))
    results.extend(_scan_torch_hub(home / ".cache" / "torch" / "hub"))
    results.extend(_scan_files(home / ".cache" / "clip", "clip", {".pt", ".bin"}))
    results.extend(_scan_files(home / ".keras" / "models", "keras", {".h5", ".keras", ".pb"}))

    _ORDER = {"transformers": 0, "timm": 1, "torch": 2, "clip": 3, "keras": 4}
    results.sort(key=lambda m: (_ORDER[m.source], m.model_id.lower()))
    return results


def print_cached_models(
    models: list[CachedModel] | None = None,
    hf_cache_dir: str | Path | None = None,
    console: Console | None = None,
) -> None:
    """Print a coloured summary of locally cached models.

    Parameters
    ----------
    models : list[CachedModel] or None
        Pre-fetched list from :func:`find_cached_models`.  If ``None``,
        calls :func:`find_cached_models` automatically.
    hf_cache_dir : str or Path or None
        Forwarded to :func:`find_cached_models` when *models* is ``None``.
    console : rich.console.Console or None
        Rich console to use for output.  Pass ``Console(record=True)`` to
        capture the output for SVG / HTML export.  Defaults to a new
        ``Console()`` that writes to stdout.
    """
    from rich.console import Console as _Console
    from rich.table import Table
    from rich.text import Text

    if models is None:
        models = find_cached_models(hf_cache_dir=hf_cache_dir)

    if console is None:
        console = _Console()

    if not models:
        console.print("No cached models found.")
        return

    def fmt_size(n: int) -> str:
        if n >= 1 << 30:
            return f"{n / (1 << 30):.1f} GB"
        if n >= 1 << 20:
            return f"{n / (1 << 20):.0f} MB"
        if n >= 1 << 10:
            return f"{n / (1 << 10):.0f} KB"
        return f"{n} B"

    table = Table(
        show_header=True,
        header_style="bold",
        box=None,
        padding=(0, 2),
        show_edge=False,
    )
    table.add_column("model_id", min_width=40)
    table.add_column("size", justify="right", min_width=8)
    table.add_column("last used", justify="right", min_width=10)

    for m in models:
        color = _RICH_COLORS.get(m.source, "white")
        table.add_row(
            Text(m.model_id, style=color),
            Text(fmt_size(m.size_bytes), style=color),
            Text(m.last_used.strftime("%Y-%m-%d"), style=color),
        )

    console.print()
    console.print(table)

    present = {m.source for m in models}
    legend_items = [
        f"[{_RICH_COLORS[src]}]██[/{_RICH_COLORS[src]}] {src}"
        for src in ("transformers", "timm", "torch", "clip", "keras")
        if src in present
    ]
    console.rule(style="dim")
    console.print("  " + "   ".join(legend_items))
