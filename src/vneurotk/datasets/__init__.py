"""Sample dataset fetchers for VneuroTK tutorials and examples.

The ``sample`` module downloads two bundled datasets from Zenodo and returns
a single root directory from which you navigate with standard path operations.

Usage
-----
>>> from vneurotk.datasets import sample
>>> root = sample.data_path()
>>> nod  = root / "nod-meg"        # NOD-MEG MEG recording
>>> mv   = root / "monkey-vision"  # MonkeyVision ephys recording

Available datasets (inside the root)
--------------------------------------
.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Sub-directory
     - Description
   * - ``nod-meg/``
     - NOD-MEG sample: 1 subject × 1 run MEG recording + 200 stimuli
   * - ``monkey-vision/``
     - MonkeyVision ephys sample: spike rasters, firing rates, stimulus responses

Examples
--------
>>> from vneurotk.datasets import sample
>>> root = sample.data_path()
>>> root / "nod-meg" / "meg"
PosixPath('.../nod-meg/meg')
"""

from vneurotk.datasets import sample

__all__ = ["sample"]
