# Dataset provenance, licensing, citation, and ethics

VneuroTK can read user-supplied neural recordings and stimuli, download sample archives, embed stimulus pixels in HDF5 files, and download model artifacts through optional backends. Technical access does not grant permission to use or redistribute any of those materials. Users must verify rights, access conditions, required citations, and ethics approvals with the authoritative provider before use.

## Datasets currently named in this repository

The repository knows only the following about its downloadable samples:

| Fetcher name | Repository-known provenance | What is not established here |
| --- | --- | --- |
| `nod-meg` | `vneurotk.datasets.sample` fetches `vneurotk-nod-meg-sample.zip` (87,243,257 bytes; SHA256 `cebcec0bab57d548c486d7e4456c1e56c5832a47e4793ddb1e8f5f6a4d403968`) from the [VneuroTK Sample Data record on Zenodo](https://zenodo.org/records/20094167), DOI [`10.5281/zenodo.20094167`](https://doi.org/10.5281/zenodo.20094167). The verified archive has 202 entries: one NOD-MEG subject/run recording, one events table, and 200 stimuli. The corresponding authoritative dataset page is [NOD-MEG `ds005810` on OpenNeuro](https://openneuro.org/datasets/ds005810), whose verified version 2.0.0 metadata names DOI [`10.18112/openneuro.ds005810.v2.0.0`](https://doi.org/10.18112/openneuro.ds005810.v2.0.0) and CC0. | The Zenodo sample record currently identifies its own license as CC BY 4.0, while OpenNeuro identifies NOD-MEG v2.0.0 as CC0. This repository does not establish which terms govern every repackaged file or stimulus. Verify both records and the files' embedded metadata before use or redistribution. |
| `monkey-vision` | The same fetcher downloads `vneurotk-monkey-vision-sample.zip` (191,862,733 bytes; SHA256 `bb5c1a8aab1faa4fb97d89adaabfa301bea7f4879b54b102f3b3a17f35ada94e`) from the directly linked [Zenodo sample record](https://zenodo.org/records/20094167). The verified archive has 9 entries, including channel-level trial raster and trial record files, for session `251024_FanFan_nsd1w_MSB`. The record currently identifies its license as CC BY 4.0. | The repository provides no authoritative upstream publication, source-dataset link, citation instructions, consent or ethics statement, or evidence that the Zenodo record's license resolves all underlying-data and stimulus rights for this sample. Verify scope and provenance with the record owner before use or redistribution. |

These identifiers and direct links make the repository's existing sources auditable; they are not new claims about authorship or rights. The Zenodo DOI identifies the sample record, not VneuroTK itself or every upstream work represented in the archives. Metadata was verified directly against the Zenodo record and OpenNeuro API; providers can revise records, so re-check them at the time of use. Do not infer that a public download, a cached copy, or inclusion in an example makes data unrestricted.

## Repository validation policy

The normal test suite validates the sample downloader and archive protections
offline with generated local archives. It does not download the published
samples. A separate weekly and manually dispatched workflow validates the real
NOD-MEG and MonkeyVision archives from trusted default-branch code. Each job
uses a fresh temporary sample directory; sample archives and extracted data are
not restored from a cache, saved to a cache, or uploaded as artifacts. This
scheduled validation is deliberately outside the normal CI aggregate and does
not change the rights, provenance, or ethics obligations described here.

## Responsibilities when adding or using data

Before downloading, processing, embedding, publishing, or sharing data:

1. Record the authoritative source URL or accession, exact version or retrieval date, checksums where available, and any transformations or subset selection.
2. Read the dataset and stimulus licenses and confirm that the intended use and redistribution are permitted. Preserve attribution, notices, and citation instructions.
3. Cite the original dataset and relevant methods publication using metadata supplied by the authoritative source. Do not fabricate a DOI or copy an unverified citation from this project.
4. Confirm that human-participant consent, ethics approval, data-use agreements, and privacy controls cover the intended work. For non-human animal data, confirm the applicable ethics and welfare approvals. VneuroTK performs no such review.
5. Remove or protect direct and indirect identifiers. Do not place restricted recordings, participant metadata, credentials, access tokens, or licensed stimuli in issues, tests, examples, or public repositories.
6. Check generated HDF5 files before sharing. A saved recording may include neural arrays, trial metadata, model provenance, and stimulus pixels; ownership and restrictions remain attached to those contents.
7. Review pretrained-model and model-weight terms separately. Backend identifiers and automatic downloads do not convey a model license or establish suitability for a scientific or clinical purpose.

When contributing a new sample or integration, provide provenance, license evidence, required citations, an ethics/privacy assessment, a minimal redistributable fixture where possible, and tests that do not require undisclosed credentials. If rights are unclear, do not add or redistribute the artifact; use a user-supplied local path and document how users can obtain it from the authoritative provider.

## Scientific and ethical limitations

VneuroTK is research software, not a clinical device. Feature extraction and file conversion can preserve biases, artifacts, and sensitive attributes in source data. Users are responsible for validating data quality, experimental assumptions, model fitness, reproducibility, and downstream interpretations. Keep immutable source data and transformation records, especially while the package and HDF5 schema remain pre-alpha.

If repository metadata conflicts with an authoritative dataset record, follow the authoritative record and report the discrepancy through a GitHub issue without attaching restricted data.
