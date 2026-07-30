"""Guards the matplotlib backend the viz test suite renders on (issue #117)."""

from __future__ import annotations

import matplotlib


def test_viz_suite_renders_on_agg() -> None:
    """The viz tests must run on Agg, never on a GUI toolkit.

    Nothing in the package pins a backend, so matplotlib picks TkAgg on Windows
    whenever tkinter imports. Every figure the suite builds then spins up a real
    Tcl/Tk interpreter, and none of them is ever displayed: there is no
    ``plt.show`` in the tests or in ``wow_forecaster/viz``. Under full-suite load
    that interpreter creation loses a race often enough to have failed four
    different viz tests between 2026-07-19 and 2026-07-29, each time with
    ``This probably means that tk wasn't installed properly`` and each time
    passing on rerun.

    Read the coverage honestly: CI runs headless Linux, where matplotlib falls
    back to Agg on its own, so this assertion holds there with or without the
    pin in ``conftest.py``. It is a real regression test on Windows and a weak
    one on CI.
    """
    assert matplotlib.get_backend().lower() == "agg"
