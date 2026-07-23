"""Durable-table backup layer.

Builds a self-contained, directly-restorable SQLite snapshot of the durable
tables (everything except the two per-observation tables) and uploads it to a
separate, no-expiry object store. See :mod:`wow_forecaster.backup.durable_backup`
and ``docs/db-backup.md``.
"""
