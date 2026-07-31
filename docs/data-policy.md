# Dataset provenance, licensing, citation, and ethics

VneuroTK can read user-supplied neural recordings and stimuli, download sample archives, embed stimulus pixels in HDF5 files, and download model artifacts through optional backends. Technical access does not grant permission to use or redistribute any of those materials. Users must verify rights, access conditions, required citations, and ethics approvals with the authoritative provider before use.

## Datasets currently named in this repository

The repository knows only the following about its downloadable samples:

| Fetcher name | Repository-known provenance | What is not established here |
| --- | --- | --- |
| `nod-meg` | `vneurotk.datasets.sample` fetches `vneurotk-nod-meg-sample.zip` (87,243,257 bytes; SHA256 `cebcec0bab57d548c486d7e4456c1e56c5832a47e4793ddb1e8f5f6a4d403968`) from Zenodo record `20094167`. The verified archive has 202 entries: one NOD-MEG subject/run recording, one events table, and 200 stimuli. The example links NOD-MEG on OpenNeuro as dataset `ds005810`. The fetcher docstring identifies the Zenodo record DOI as `10.5281/zenodo.20094167`. | This repository does not document the sample archive's license, the relationship between every repackaged file and the full OpenNeuro dataset, or the redistribution terms for its stimuli. Verify the Zenodo and OpenNeuro records and the files' embedded metadata before use or redistribution. |
| `monkey-vision` | The same fetcher downloads `vneurotk-monkey-vision-sample.zip` (191,862,733 bytes; SHA256 `bb5c1a8aab1faa4fb97d89adaabfa301bea7f4879b54b102f3b3a17f35ada94e`) from Zenodo record `20094167`. The verified archive has 9 entries, including channel-level trial raster and trial record files, for session `251024_FanFan_nsd1w_MSB`. | The repository provides no authoritative upstream publication, license, citation instructions, consent or ethics statement, or redistribution grant for this sample. Treat those facts as unknown until verified with the record owner or authoritative source. |

These identifiers are reported because they already appear in the code and documentation. They are not new claims about authorship or licensing. The Zenodo DOI above identifies the sample record, not VneuroTK. Do not infer that a public download, a cached copy, or inclusion in an example makes data unrestricted.

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
