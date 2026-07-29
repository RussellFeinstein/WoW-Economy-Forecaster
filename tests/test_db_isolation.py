"""Guard: the suite never resolves to the production database (issue #113).

The ``isolated_product_db`` autouse fixture in ``tests/conftest.py`` is what
stops a test that builds a real pipeline stage from writing into
``data/db/wow_forecaster.db``. That protection is invisible in normal runs, so
without these tests a later refactor could drop the fixture and nothing would
fail. The original leak was silent for weeks for exactly that reason.
"""

from __future__ import annotations

import os
from pathlib import Path

from wow_forecaster.config import load_config

PRODUCTION_DB = "data/db/wow_forecaster.db"


def _posix(path: str) -> str:
    """Compare paths without caring which slash Windows handed us."""
    return path.replace("\\", "/")


def test_load_config_does_not_resolve_to_the_production_database() -> None:
    """The config a test sees must not point at the real database."""
    resolved = _posix(load_config().database.db_path)
    assert not resolved.endswith(PRODUCTION_DB), (
        f"load_config() resolved to the production database ({resolved}). "
        "The isolated_product_db fixture in tests/conftest.py is missing or "
        "has been overridden."
    )


def test_the_override_env_var_is_set_for_every_test() -> None:
    """The fixture is autouse, so this test gets it without asking."""
    assert os.environ.get("WOW_FORECASTER_DB_PATH")


def test_the_isolated_path_lives_under_tmp(isolated_product_db: Path) -> None:
    """The fixture hands back the path it pinned, and it is a tmp path."""
    assert str(isolated_product_db) == os.environ["WOW_FORECASTER_DB_PATH"]
    assert not _posix(str(isolated_product_db)).endswith(PRODUCTION_DB)


def test_a_test_may_still_choose_its_own_database(monkeypatch, tmp_path: Path) -> None:
    """The guard sets a default; it does not take the choice away."""
    chosen = tmp_path / "chosen.db"
    monkeypatch.setenv("WOW_FORECASTER_DB_PATH", str(chosen))
    assert load_config().database.db_path == str(chosen)
