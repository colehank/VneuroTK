"""pytest configuration — bridges loguru to pytest's caplog."""

from __future__ import annotations

import importlib.util
import logging
import os

import pytest
from loguru import logger

_MARKER_ENV = {
    "vision": "VNEUROTK_TEST_VISION",
    "viz": "VNEUROTK_TEST_VIZ",
    "backend_transformers": "VNEUROTK_TEST_BACKEND_TRANSFORMERS",
    "backend_timm": "VNEUROTK_TEST_BACKEND_TIMM",
    "backend_thingsvision": "VNEUROTK_TEST_BACKEND_THINGSVISION",
    "hdf5_compat": "VNEUROTK_TEST_HDF5_COMPAT",
    "sample_data": "VNEUROTK_TEST_SAMPLE_DATA",
    "integration": "VNEUROTK_TEST_INTEGRATION",
    "network": "VNEUROTK_RUN_NETWORK",
    "slow": "VNEUROTK_TEST_SLOW",
}

_MARKER_IMPORT = {
    "vision": "torch",
    "viz": "matplotlib",
    "backend_transformers": "transformers",
    "backend_timm": "timm",
    "backend_thingsvision": "thingsvision",
}


def pytest_collection_modifyitems(config, items):
    """Keep lanes opt-in and reject configured lanes missing dependencies."""
    enabled_markers = {marker for marker, env_var in _MARKER_ENV.items() if os.environ.get(env_var) == "1"}
    required_imports = {
        module
        for marker, module in _MARKER_IMPORT.items()
        if marker in enabled_markers and any(item.get_closest_marker(marker) is not None for item in items)
    }
    missing = [module for module in required_imports if not _can_import(module)]
    if missing:
        raise pytest.UsageError(f"configured test lane is missing required dependencies: {', '.join(sorted(missing))}")

    for item in items:
        for marker, env_var in _MARKER_ENV.items():
            if item.get_closest_marker(marker) is not None and marker not in enabled_markers:
                item.add_marker(pytest.mark.skip(reason=f"set {env_var}=1 to run {marker} tests"))


def _can_import(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


@pytest.fixture(autouse=True)
def _propagate_loguru_to_caplog(caplog):
    """Enable vneurotk logging and forward loguru messages to pytest caplog."""
    logger.enable("vneurotk")
    caplog.set_level(logging.DEBUG)

    def _sink(message):
        record = message.record
        level = getattr(logging, record["level"].name, logging.DEBUG)
        logging.getLogger(record["name"]).log(level, record["message"])

    handler_id = logger.add(_sink, level="DEBUG")
    yield
    logger.remove(handler_id)
    logger.disable("vneurotk")
