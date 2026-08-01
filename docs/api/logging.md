# Logging

VneuroTK does not configure logging when imported. It uses [Loguru](https://loguru.readthedocs.io/), so records remain available to sinks that the host application has already configured, without VneuroTK deleting or replacing those sinks.

To add a VneuroTK-owned sink explicitly:

```python
import vneurotk as vtk

vtk.setup_logging("INFO")
```

`setup_logging()` is idempotent: another call replaces only the sink created by the previous call. Its sink is filtered to the `vneurotk` package. `set_log_level("DEBUG")` is a convenience API for configuring that VneuroTK sink; with no argument it reads `VNTK_LOGGING_LEVEL` and defaults to `INFO`.

MNE logging and Python warning filters are process-global, so VneuroTK leaves them unchanged by default. Applications that deliberately want VneuroTK to configure them must opt in:

```python
vtk.setup_logging(
    "INFO",
    mne_level="ERROR",
    suppress_mne_naming_warnings=True,
)
```

## API

```{eval-rst}
.. autofunction:: vneurotk._log.setup_logging
```
```{eval-rst}
.. autofunction:: vneurotk._log.set_log_level
```