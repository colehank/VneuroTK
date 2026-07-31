# HDF5 recording format

!!! warning "Pre-alpha schema"
    The current writer uses schema 1, but the format is not yet declared stable.
    Keep original source data and pin VneuroTK when files are part of a
    reproducible workflow.

VneuroTK recordings saved by `BaseData.save()` use HDF5. The root attributes
identify the file and select the reader before recording data is interpreted.

## Format header

Schema 1 writers set these root attributes:

| Attribute | Schema 1 value | Purpose |
| --- | --- | --- |
| `vneurotk_format` | `"recording"` | File-type magic. Other values are rejected. |
| `vneurotk_schema_version` | integer `1` | On-disk recording schema version. |
| `writer_version` | package version string | VneuroTK version that wrote the file. |
| `data_mode` | nonempty `"continuous"`, `"epochs"`, or `"patterns"` | Neural/trial layout semantics. |

Readers in this release support schema versions **0 through 1**. A file with
neither `vneurotk_format` nor `vneurotk_schema_version` is treated as historical schema
0. A partial header, wrong magic, malformed version, or version outside the
supported range is rejected rather than guessed.

`writer_version` records provenance rather than selecting the reader, but it is
a required schema-1 header field. Compatibility is determined by
`vneurotk_schema_version`.

## Compatibility policy

Schema 0 is the unversioned format written before the header was introduced.
Its normalization is isolated from the current schema reader. Compatibility
includes:

- dense neural arrays (`neuro`);
- COO neural arrays (`neuro_row`, `neuro_col`, `neuro_data`, plus shape and
  dtype attributes);
- numeric and UTF-8 string stimulus labels;
- the historical `data_mode="continues"` typo, normalized to `"continuous"`;
- historical image entries with `kind="path"`, including relative paths
  resolved from the recording's directory;
- trial arrays, trial metadata, image databases, and visual representations.

New files are always written as schema 1. Loading a schema-0 file does not
modify it. VneuroTK does not currently provide in-place migration.

## Schema 1 layout

Neural data is stored either as a dense `neuro` dataset or flattened COO
components selected by the `neuro_format` root attribute. Recording metadata
uses the `neuro_info`, `vision_info`, `trial_info`, and `trial_meta` groups.
Trial arrays (`trial`, `trial_starts`, `trial_ends`, and `vision_onsets`) and
stimulus labels are root datasets when present. Optional images live under
`stimuli_db`; optional extracted features live under `vision_store`.

Neural arrays and visual-representation arrays are chunked and gzip-compressed
(level 4) by default so lazy readers can fetch individual datasets efficiently.
Each schema-1 `vision_store/<record>` group also carries an
`extraction_provenance` attribute: deterministic JSON for
`ExtractionProvenance` serialization version 1. It records backend/model and
locally available revision, pretrained state, processor/preprocessing,
selector, dependency versions, dtype/device, VneuroTK writer version, and an
optional stimulus content hash. Provenance is stored per feature record so
subset extraction and overwrite retain the metadata of the array they actually
store. Historical schema-0 visual records have no such attribute and load with
their model ID preserved while unavailable provenance fields are explicitly
`"unknown"`.

`BaseData.save()` accepts `compression`, `compression_opts`, and
`chunk_target_bytes` to change the filter or chunk target, including
`compression=None` to disable compression. Pass `pre_load=True` to
`vneurotk.read()` to materialize neural data eagerly.

## Provenance boundary

The schema records **writer provenance**, not a guarantee that an environment
can be recreated. Root `writer_version` identifies the VneuroTK writer. Each
vision record's `extraction_provenance` captures locally discoverable backend,
model, preprocessing, selector, dependency, dtype, and device metadata plus an
optional caller-supplied stimulus digest. Unknown values remain `"unknown"`.
VneuroTK does not fetch registry metadata, hash stimuli, or embed model weights
while saving. Callers should store environment lock files and stimulus/model
artifacts separately when exact reproducibility is required.

## Safe scalar identifiers

Schema 1 never derives an HDF5 path from a stimulus ID. Image database entries
are stored under ordered numeric groups and each ID is encoded separately with
an explicit scalar type. Supported IDs are `bool`, `int`, finite `float`, and UTF-8
`str`; this preserves distinctions such as `1` versus `"1"` and safely handles
slashes and arbitrary Unicode text. Non-finite float IDs (`NaN` and infinities)
and unsupported ID values fail the save before
the destination is replaced. The same typed scalar encoding is used for object
stimulus labels and visual-representation IDs; missing object labels may use
`None`, `pandas.NA`, or `pandas.NaT`, but arbitrary objects are rejected rather
than converted to empty strings.

## Trial metadata

Schema 1 trial metadata stores column labels, the columns index type and name,
row index values and names, and dtype metadata separately from values. It
round-trips NumPy numeric/boolean/string
dtypes, pandas nullable integer/float/boolean/string dtypes, categoricals
(including ordering and naive or timezone-aware datetime categories), and standalone
naive or timezone-aware datetime columns.
Object columns are accepted only when every value is one of the supported
scalar or missing types. Unsupported extension dtypes or arbitrary Python
objects fail clearly instead of being stringified.

## Atomic writes

Saves are written to a temporary file in the destination directory, flushed,
closed, reopened for basic header/layout validation, and installed with
`os.replace`. Any encoding, write, validation, or replacement failure leaves an
existing destination unchanged and removes the temporary file. Overwriting
preserves the destination's permission bits; newly created files receive the
normal process-umask-respecting mode. Lazy neuro, image, and activation readers
capture the backing file identity and fail explicitly if the pathname is later
atomically replaced, preventing data from different snapshots from being mixed.
