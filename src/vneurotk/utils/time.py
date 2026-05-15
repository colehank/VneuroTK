"""
General-purpose signal time utilities.
"""

from __future__ import annotations


def sec_to_samples(sec: float, sfreq: float) -> int:
    """Convert time in seconds to sample count.

    Parameters
    ----------
    sec : float
        Time in seconds.
    sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    int
        Number of samples.
    """
    return int(round(sec * sfreq))


def samples_to_sec(samples: int, sfreq: float) -> float:
    """Convert sample count to time in seconds.

    Parameters
    ----------
    samples : int
        Number of samples.
    sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    float
        Time in seconds.
    """
    return samples / sfreq
