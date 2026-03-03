"""Conftest for CLI smoke tests.

``load_config()`` (called by every CLI command via ``_load_config_or_exit()``)
invokes ``load_dotenv()``, which writes credentials from ``.env`` directly
into ``os.environ``.  Those additions persist across the test session and can
cause unrelated tests (e.g. IngestStage) to switch from fixture mode to live
API mode unexpectedly.

This autouse fixture snapshots and fully restores ``os.environ`` after each
CLI test so that dotenv-loaded values do not leak beyond this package.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _restore_env():
    """Restore os.environ to its pre-test state after each CLI smoke test."""
    snapshot = os.environ.copy()
    yield
    # Remove any keys added during the test (e.g. by load_dotenv).
    for key in list(os.environ.keys()):
        if key not in snapshot:
            del os.environ[key]
    # Restore changed or deleted keys.
    for key, val in snapshot.items():
        os.environ[key] = val
