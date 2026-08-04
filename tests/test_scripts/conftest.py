"""Shared expectations for the scheduled-task .bat scripts.

The four scheduled scripts (run_hourly, run_daily, run_healthcheck, run_backup)
each append to their own log under logs/, and each opens a run's entry with the
same bare rule.  The constant lives here rather than in four test modules so the
convention has one definition: a script that drifts off it fails its own test,
and a deliberate change to the rule is a one-line edit here plus the four bats.
"""

from __future__ import annotations

# Written by each script as the first line of every entry, on a line of its own.
# Bare on purpose: these logs previously opened with "[date time] ====", which
# was indistinguishable at a glance from the timestamped lines around it, so the
# start of a run was hard to find in a file months of runs had appended to.
ENTRY_SEPARATOR = "=" * 89
