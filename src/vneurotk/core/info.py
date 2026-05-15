"""Info summary object for VneuroTK data containers."""

from __future__ import annotations

from typing import Any

_STYLE = (
    "<style scoped>"
    ".vtk-info{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',"
    "Roboto,sans-serif;font-size:13px;max-width:480px}"
    ".vtk-info summary{cursor:pointer;padding:6px 0;font-size:14px}"
    ".vtk-info table{width:100%;border-collapse:collapse;margin:0 0 8px 0}"
    ".vtk-info th,.vtk-info td{padding:4px 12px;border-bottom:1px solid currentColor;border-bottom-opacity:0.2}"
    ".vtk-info th{text-align:left;width:50%;font-weight:500;opacity:0.75}"
    ".vtk-info td{text-align:right;width:50%}"
    ".vtk-info tr:last-child th,.vtk-info tr:last-child td{border-bottom:none}"
    ".vtk-info .vtk-na{opacity:0.5;font-style:italic}"
    "</style>"
)


class Info:
    """Summary object returned by :attr:`BaseData.info`.

    Parameters
    ----------
    neuro : dict
        Dict with keys ``n_time``, ``n_neuro``, ``sfreq``, ``highpass``,
        ``lowpass``.
    visual : dict or None
        Dict with key ``n_stim``.
    trial : dict or None
        Dict with keys ``baseline``, ``trial_window``.
    configured : bool
        Whether the parent :class:`BaseData` has been configured.
    data_mode : str
        ``"continuous"``, ``"epochs"``, or ``"patterns"``.
    """

    def __init__(
        self,
        neuro: dict[str, Any],
        visual: dict[str, Any] | None,
        trial: dict[str, Any] | None,
        configured: bool,
        data_mode: str = "continuous",
    ) -> None:
        self._neuro = neuro
        self._visual = visual
        self._trial = trial
        self._configured = configured
        self._data_mode = data_mode

    # --- HTML helpers ---------------------------------------------------

    @staticmethod
    def _table(rows: list[tuple[str, str]]) -> str:
        trs = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
        return f"<table>{trs}</table>"

    @staticmethod
    def _section(title: str, body: str) -> str:
        return f"<details open><summary><strong>{title}</strong></summary>{body}</details>"

    @staticmethod
    def _na(text: str = "Not configured") -> str:
        return f'<span class="vtk-na">{text}</span>'

    # --- public repr ------------------------------------

    def _repr_html_(self) -> str:
        n = self._neuro
        sfreq = n.get("sfreq")
        hp = n.get("highpass")
        lp = n.get("lowpass")
        neuro_rows = [
            ("Time points", str(n["n_time"])),
            ("Channels", str(n["n_chan"])),
            ("Sampling frequency", f"{sfreq:.2f} Hz" if sfreq is not None else "N/A"),
            ("Highpass", f"{hp:.2f} Hz" if hp is not None else "N/A"),
            ("Lowpass", f"{lp:.2f} Hz" if lp is not None else "N/A"),
            ("Data mode", self._data_mode),
        ]
        parts = [self._section("Neuro", self._table(neuro_rows))]

        if self._configured and self._visual is not None:
            parts.append(
                self._section(
                    "Vision",
                    self._table([("n_visual", str(self._visual["n_stim"]))]),
                )
            )
        else:
            parts.append(
                self._section(
                    "Vision",
                    self._table([("Status", self._na())]),
                )
            )

        if self._configured and self._trial is not None:
            t = self._trial
            parts.append(
                self._section(
                    "Trial",
                    self._table(
                        [
                            ("Baseline", str(t["baseline"])),
                            ("Trial window", str(t["trial_window"])),
                        ]
                    ),
                )
            )
        else:
            parts.append(
                self._section(
                    "Trial",
                    self._table([("Status", self._na())]),
                )
            )

        body = "".join(parts)
        return f'{_STYLE}<div class="vtk-info">{body}</div>'

    def __repr__(self) -> str:
        n = self._neuro
        sfreq = n.get("sfreq")
        hp = n.get("highpass")
        lp = n.get("lowpass")
        lines = [
            "Info",
            f"  Neuro: Time points={n['n_time']}, Channels={n['n_chan']}, "
            f"sfreq={sfreq}, highpass={hp}, lowpass={lp}, "
            f"data_mode={self._data_mode}",
        ]
        if self._configured and self._visual is not None:
            lines.append(f"  Vision: n_vision={self._visual['n_stim']}")
        else:
            lines.append("  Vision: Not configured")
        if self._configured and self._trial is not None:
            t = self._trial
            lines.append(f"  Trial: baseline={t['baseline']}, trial_window={t['trial_window']}")
        else:
            lines.append("  Trial: Not configured")
        return "\n".join(lines)
