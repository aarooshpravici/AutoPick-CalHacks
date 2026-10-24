"""Shared pytest setup. Slow tests (marked @pytest.mark.slow: they train a model) are skipped
unless RUN_SLOW=1, so the normal suite stays fast: RUN_SLOW=1 pytest -q -m slow runs them."""

import os

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: trains a model or runs the whole pipeline; only with RUN_SLOW=1")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RUN_SLOW") == "1":
        return
    skip = pytest.mark.skip(reason="slow: set RUN_SLOW=1 to run it")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
