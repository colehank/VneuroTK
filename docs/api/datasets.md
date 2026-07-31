# Datasets

Sample dataset fetchers for tutorials and examples.

!!! warning "Verify rights and citations"
    Downloading a sample does not establish permission to use or redistribute it.
    Review the [data policy](../data-policy.md), then verify license, citation, and
    ethics requirements against each authoritative dataset record.

## Download and cache behavior

`sample.data_path()` downloads from Zenodo record `20094167`. Each archive is
bounded to its authoritative byte size while streaming and rejected if it is
oversized or truncated, then verified against its pinned SHA256 digest before
extraction. Downloads use Pooch's requests-based HTTP downloader with a
10-second connection timeout, a 60-second read timeout, and at most two retries
after the first attempt. Extraction rejects unsafe paths, links, unexpected
dataset roots, and archives outside documented resource limits. A SHA256-bound
completion marker makes an already completed extraction reusable; an incomplete
extraction is never installed as the dataset tree.

Passing `path=None` selects Pooch's platform cache. Any other value is handled
as an explicit path, including an empty path. Dataset selections must be a
known name or a non-empty, duplicate-free list of known names.

## Validation

The downloader, digest verification, safe extraction, completion markers, and
resource limits are tested offline with generated archives in the normal test
suite; those tests do not fetch a published sample. A separate weekly and
manually dispatched workflow downloads each published archive into a fresh,
job-local temporary cache and runs bounded NOD-MEG and MonkeyVision smoke tests.
The scheduled workflow is intentionally separate from the normal CI aggregate,
and it neither caches nor uploads sample data or archives.

For local offline validation, run `just test-sample-integrity`; it runs only
`tests/test_datasets_sample.py` and enables none of the sample-data, network,
integration, or slow gates.

Each real-data recipe includes both the matching archive integrity test and its
bounded smoke tests (`-k nod` or `-k monkey`). These lanes are explicitly
network-, sample-data-, integration-, and slow-gated; use
`just test-sample-nod` or `just test-sample-monkey` only when a real cached or
fresh sample is available. `just test-samples` aggregates the offline contract,
NOD-MEG, and MonkeyVision recipes.

## Extracted layout

The NOD archive has 202 entries: one cleaned MEG recording, one events table,
and 200 stimuli under `nod-meg/`. The MonkeyVision archive has 9 entries under
`monkey-vision/sessions/251024_FanFan_nsd1w_MSB/`:

- `TrialRaster_251024_FanFan_nsd1w_MSB.h5`
- `TrialRecord_251024_FanFan_nsd1w_MSB.csv`
- `MeanFr_251024_FanFan_nsd1w_MSB.h5`
- `ChMeanFr_251024_FanFan_nsd1w_MSB.h5`
- `ChStimFr_251024_FanFan_nsd1w_MSB.h5`
- `ChTrialRaster_251024_FanFan_nsd1w_MSB.h5`
- `ChTrialRecord_251024_FanFan_nsd1w_MSB.csv`
- `UnitProp_251024_FanFan_nsd1w_MSB.csv`
- `ChProp_251024_FanFan_nsd1w_MSB.csv`

::: vneurotk.datasets.sample
