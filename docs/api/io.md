# I/O

Path classes and data loaders for VneuroTK.

## Path Classes

```{eval-rst}
.. autoclass:: vneurotk.io.path.EphysPath
   :members:
```
```{eval-rst}
.. autoclass:: vneurotk.io.path.MNEPath
   :members:
```
```{eval-rst}
.. autoclass:: vneurotk.io.path.VTKPath
   :members:
```
```{eval-rst}
.. autoclass:: vneurotk.io.path.BIDSPath
   :members:
```
## Path constants

```{eval-rst}
.. autodata:: vneurotk.io.path.EPHYS_DTYPES
```
```{eval-rst}
.. autodata:: vneurotk.io.path.EPHYS_EXTENSIONS
```

## Reading Data

```{eval-rst}
.. autofunction:: vneurotk.io.loader.read
```
```{eval-rst}
.. autoclass:: vneurotk.io.loader.LazyNeuroLoader
   :members:
```
```{eval-rst}
.. autoclass:: vneurotk.io.loader.LazyH5Dict
   :members:
```