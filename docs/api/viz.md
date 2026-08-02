# Visualization

Matplotlib plots for inspecting stimulus timing and neural activity. This API is separate from [`vneurotk.vision`](vision.md), which extracts DNN representations from images. Install plotting support with `vneurotk[viz]`; see the [visualization guide](../usage/viz) for runnable examples.

## Supported data and units

`BaseData.plot` supports `continuous` recordings and pre-epoched `epochs`; epochs are flattened across trial and time for display. It rejects `patterns` because aggregated pattern rows have no time axis. Neural data and a positive, finite `neuro_info["sfreq"]` must be available. Configured trial metadata is optional: an unconfigured time-series can still show neural activity, while configured stimulus labels, trial IDs, and trial windows add the trial-setting panel.

For both public entry points, display-window bound types determine units:

- integral bounds, including NumPy integer scalars, are sample indices;
- non-integral real bounds, including values such as `2.0`, are seconds and are converted with `sfreq`;
- the two finite bounds must be ordered and overlap the recording.

This makes `2` sample 2, but `2.0` two seconds. A trial window follows the same integer-samples/non-integral-real-seconds convention.

## `BaseData.plot`

The convenience method uses the recording's neural data, sampling frequency, labels, trial IDs, and trial window. It returns a caller-owned Matplotlib `Figure` for annotation, saving, or closing. Plotting requires Matplotlib from the `viz` extra.

```{eval-rst}
.. automethod:: vneurotk.core.recording.BaseData.plot
   :no-index:
```

## `plot_data`

Use the lower-level function for arrays that are not wrapped in `BaseData`. `neuro` must be a nonempty two-dimensional array shaped `(n_samples, n_channels)`; `visual` must be a one-dimensional label array with one value per sample; and `sfreq` must be positive and finite. Null visual values represent no stimulus.

`trial` and `trial_window` are optional but must be supplied together. `trial` must contain one null or nonnegative integer trial ID per sample. `trial_window` may be one shared ordered pair or one pair per referenced trial ID. Trial windows must span at least one sample.

```{eval-rst}
.. autofunction:: vneurotk.viz.data.plot_data
```

See the [visualization example](../example_ipynb/viz) for a complete synthetic evoked-response workflow and the [DNN vision API](vision.md) when the goal is image feature extraction rather than plotting.
