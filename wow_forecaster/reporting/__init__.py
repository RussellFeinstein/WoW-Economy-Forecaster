"""
wow_forecaster.reporting — Local report reading, formatting, and export.

This package reads already-persisted output files (JSON, CSV) and formats
them for CLI display or flat-file export (Power BI / manual analysis).

It does NOT produce new data — all reads are from data/outputs/.

Modules:
  reader     — File discovery, loading helpers, freshness checks.
  formatters — ASCII terminal table formatters for Typer CLI commands.
  export     — CSV/JSON flat-file export helpers.
"""
