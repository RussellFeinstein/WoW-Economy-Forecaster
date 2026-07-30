"""Pins the matplotlib backend for the viz tests (issue #117).

Pytest imports a directory's ``conftest.py`` before the test modules in it, so
this runs before anything here imports pyplot and locks in a backend.

Agg is what headless CI already falls back to. Without the pin, Windows picks
TkAgg instead and every figure the suite builds spins up a Tcl/Tk interpreter
that no test ever displays, which flakes under full-suite load. Scoped to this
directory because these are the only modules in the suite that import
matplotlib, and a root-level pin would have to guard the import to keep the
rest of the suite runnable without the ``[viz]`` extra.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
