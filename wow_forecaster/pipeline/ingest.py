"""
IngestStage — raw data ingestion into ``market_observations_raw``.

Current status: **STUB** — ``_execute()`` raises ``NotImplementedError``.

Future implementation:
  1. Replace ``_fetch_source()`` to call Blizzard AH API or read TSM export.
  2. Validate each row as a ``RawMarketObservation``.
  3. Bulk-insert via ``MarketObservationRepository.insert_raw_batch()``.
  4. Return count of rows inserted.

Extension points:
  - ``_fetch_source(source_path)`` → override for different data sources.
  - ``_validate_record(raw_dict)`` → override for source-specific cleaning.
"""

from __future__ import annotations

from wow_forecaster.models.meta import RunMetadata
from wow_forecaster.pipeline.base import PipelineStage


class IngestStage(PipelineStage):
    """Fetch raw AH data from a source and write to ``market_observations_raw``.

    Stub: not yet implemented. Raises ``NotImplementedError`` when run.
    """

    stage_name = "ingest"

    def _execute(self, run: RunMetadata, source_path: str | None = None, **kwargs) -> int:
        """Ingest raw observations from ``source_path`` into the database.

        Args:
            run: In-progress ``RunMetadata`` (mutable).
            source_path: Path to raw data file (JSON/CSV). If ``None``, uses
                the configured default raw directory.

        Returns:
            Number of raw observation rows inserted.

        Raises:
            NotImplementedError: This stage is not yet implemented.
        """
        raise NotImplementedError(
            "IngestStage is not yet implemented. "
            "Implement _fetch_source() to connect to a real data source."
        )

    def _fetch_source(self, source_path: str | None) -> list[dict]:
        """Fetch raw records from a data source.

        Replace this method with a real data source connector.

        Args:
            source_path: Path to a local file, or ``None`` for the default source.

        Returns:
            List of raw record dicts to be validated as ``RawMarketObservation``.

        Raises:
            NotImplementedError: Stub.
        """
        raise NotImplementedError("_fetch_source() is a stub. Implement for your data source.")
