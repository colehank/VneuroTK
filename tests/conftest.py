"""pytest configuration — bridges loguru to pytest's caplog."""

from __future__ import annotations

import logging

import pytest
from loguru import logger


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
