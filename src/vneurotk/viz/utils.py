import matplotlib.ticker as ticker
import numpy as np


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------
def _truncate_label(label: str, maxlen: int = 3) -> str:
    """Shorten a label string for axis display."""
    if label is None:
        return "none"
    s = str(label)
    return f"{s[:maxlen]}.." if len(s) > maxlen else s


def _is_null(v) -> bool:
    """Check if a label value is null/missing."""
    return v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip().upper() in ("NONE", "NAN", "")


# ---------------------------------------------------------------------------
# Tick formatting
# ---------------------------------------------------------------------------
def _smart_ticks(vmin: float, vmax: float, force_int: bool = False):
    """Generate boundary-inclusive tick values and a formatter.

    Returns
    -------
    ticks : list of float
        Tick positions including *vmin* and *vmax*.
    formatter : ticker.Formatter
        Matplotlib formatter for these ticks.
    """
    auto = ticker.MaxNLocator(nbins=4, min_n_ticks=2).tick_values(vmin, vmax)
    margin = (vmax - vmin) * 0.15
    internal = [t for t in auto if (vmin + margin) < t < (vmax - margin)]

    if force_int:
        ticks = sorted(set([int(vmin)] + [int(t) for t in internal] + [int(vmax)]))
    else:
        ticks = [vmin] + internal + [vmax]

    # Scientific notation for very large / very small values
    max_abs = max(abs(vmin), abs(vmax))
    if not force_int and max_abs > 0 and (max_abs < 0.01 or max_abs >= 10000):
        exponent = int(np.floor(np.log10(max_abs)))
        scale = 10**exponent
        offset_str = f"1e{exponent}"
    else:
        scale = 1.0
        offset_str = None

    def _fmt(x, _pos):
        if force_int:
            return f"{int(x)}"
        val = x / scale
        return "0.0" if val == 0 else f"{val:.1f}"

    return ticks, ticker.FuncFormatter(_fmt), offset_str
