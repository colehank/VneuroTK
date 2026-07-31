# Historical HDF5 fixtures

These files are immutable compatibility inputs captured for the schema-v1
introduction. They were generated once with `h5py` to reproduce the unversioned
layout written by VneuroTK before schema metadata existed. Tests read these
checked-in binaries and must not regenerate them.

- `v0_dense_numeric.h5`: dense numeric continuous recording, historical
  `data_mode="continues"`, array images, and mixed trial metadata.
- `v0_dense_string_path.h5`: dense continuous recording with string stimulus
  labels and a historical `stimuli_db` entry using `kind="path"`.
- `v0_coo_epochs_vision.h5`: COO epochs recording with a stored visual
  representation and trial metadata.
- `v0_path_image.png`: relative image target referenced by the string fixture.

`SHA256SUMS` is the authoritative integrity manifest. If a fixture must change,
explain the historical format change in this file and update the manifest in the
same review.
