"""Tests for the optional visualization feature stack."""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

pytestmark = [
    pytest.mark.viz,
    pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None, reason="matplotlib is not installed"),
]


def test_plot_data_returns_figure():
    from matplotlib import pyplot as plt
    from matplotlib.figure import Figure

    from vneurotk.viz import plot_data

    neuro = np.arange(40, dtype=float).reshape(20, 2)
    visual = np.array([None] * 5 + ["stim"] * 5 + [None] * 10, dtype=object)

    figure = plot_data(neuro, visual, sfreq=10.0, window=(0, 20))

    assert isinstance(figure, Figure)
    assert len(figure.axes) >= 2
    plt.close(figure)


def test_unconfigured_continuous_plot_has_no_stimulus_category():
    from matplotlib import pyplot as plt

    from vneurotk import BaseData

    data = BaseData.for_continuous(np.ones((20, 2)), {"sfreq": 10.0})

    figure = data.plot(window=(0, 20))

    assert [tick.get_text() for tick in figure.axes[0].get_yticklabels()] == ["none"]
    plt.close(figure)


def test_pattern_data_cannot_be_plotted_as_a_recording():
    from vneurotk import BaseData

    data = BaseData.for_patterns(np.ones((4, 2)))

    with pytest.raises(ValueError, match="no time axis"):
        data.plot()


def test_plot_data_validates_aligned_inputs_and_sampling_frequency():
    from vneurotk.viz import plot_data

    neuro = np.ones((20, 2))
    labels = np.full(20, None, dtype=object)

    with pytest.raises(ValueError, match="two-dimensional"):
        plot_data(neuro[:, 0], labels, sfreq=10.0)
    with pytest.raises(ValueError, match="same number of samples"):
        plot_data(neuro, labels[:-1], sfreq=10.0)
    with pytest.raises(ValueError, match="positive and finite"):
        plot_data(neuro, labels, sfreq=0.0)


def test_plot_data_validates_display_window():
    from vneurotk.viz import plot_data

    neuro = np.ones((20, 2))
    labels = np.full(20, None, dtype=object)

    with pytest.raises(ValueError, match="start before end"):
        plot_data(neuro, labels, sfreq=10.0, window=(10, 5))
    with pytest.raises(ValueError, match="does not contain any samples"):
        plot_data(neuro, labels, sfreq=10.0, window=(30, 40))


def test_plot_data_requires_paired_trial_annotations():
    from vneurotk.viz import plot_data

    neuro = np.ones((20, 2))
    labels = np.full(20, None, dtype=object)
    trial = np.full(20, None, dtype=object)

    with pytest.raises(ValueError, match="provided together"):
        plot_data(neuro, labels, sfreq=10.0, trial=trial)


def test_plot_data_with_trial_annotations():
    from matplotlib import pyplot as plt

    from vneurotk.viz import plot_data

    neuro = np.ones((20, 2))
    visual = np.array([None] * 7 + ["stim"] + [None] * 12, dtype=object)
    trial = np.array([None] * 5 + [1] * 5 + [None] * 10, dtype=object)

    figure = plot_data(neuro, visual, sfreq=10.0, trial=trial, trial_window=[-2, 3], window=(0, 20))

    assert figure.axes[0].get_title(loc="left") == "Trial setting"
    plt.close(figure)


def test_epochs_varying_onsets_plot_relative_to_each_trial_onset():
    from matplotlib import pyplot as plt

    from vneurotk import BaseData

    data = BaseData.for_epochs(np.ones((3, 10, 2)), {"sfreq": 10.0})
    data.configure(stim_ids=["a", "b", "c"], vision_onsets=np.array([2, 4, 7]))

    figure = data.plot(window=(0, 30))

    stimulus_axis = figure.axes[0]
    trial_points = stimulus_axis.collections[-1]
    relative_times = np.asarray(trial_points.get_array())
    np.testing.assert_allclose(
        relative_times,
        np.concatenate(
            [
                np.arange(-2, 8) / 10,
                np.arange(-4, 6) / 10,
                np.arange(-7, 3) / 10,
            ]
        ),
    )
    plt.close(figure)


def test_plot_data_accepts_numpy_float_window_as_seconds():
    from matplotlib import pyplot as plt

    from vneurotk.viz import plot_data

    neuro = np.ones((20, 2))
    visual = np.full(20, None, dtype=object)

    figure = plot_data(neuro, visual, sfreq=10.0, window=(np.float64(0.2), np.float64(0.8)))

    neural_axis = next(axis for axis in figure.axes if axis.get_title(loc="left") == "Neural Activity")
    image = neural_axis.images[0]
    assert np.asarray(image.get_array()).shape == (2, 6)
    plt.close(figure)
