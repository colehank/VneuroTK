"""Optional visualization utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["plot_data"]


def __getattr__(name: str) -> Any:
    """Load Matplotlib-backed helpers only when requested."""
    if name == "plot_data":
        try:
            from vneurotk.viz.data import plot_data
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith("matplotlib"):
                raise ImportError(
                    "Visualization requires Matplotlib. Install it with: pip install 'vneurotk[viz]'"
                ) from exc
            raise
        return plot_data
    raise AttributeError(f"module 'vneurotk.viz' has no attribute {name!r}")


if TYPE_CHECKING:
    from vneurotk.viz.data import plot_data  # noqa: F401
