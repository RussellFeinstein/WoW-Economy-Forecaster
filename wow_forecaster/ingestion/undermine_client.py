"""
Undermine Exchange client — typed stub with fixture data.

API:   https://undermine.exchange  (Nexus Hub data service)
Docs:  https://nexushub.co/developers (API documentation when available)

Credential setup (.env, gitignored):
  UNDERMINE_API_KEY=your_key_here

When API access is available:
  1. Add ``httpx>=0.27`` to pyproject.toml dependencies.
  2. Implement ``fetch_realm_auctions()`` and ``fetch_market_values()`` below
     by replacing the NotImplementedError bodies with real HTTP calls.
  3. Set USE_UNDERMINE_FIXTURE=0 in .env to switch out of stub mode.

IngestStage checks os.environ.get("UNDERMINE_API_KEY") to decide whether to
call ``fetch_*`` (real) or ``get_fixture_response()`` (stub).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import ClassVar, Optional

logger = logging.getLogger(__name__)


# ── Response types ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class UndermineAuctionRecord:
    """A single AH listing as returned by Undermine Exchange."""

    item_id: int
    realm_slug: str
    faction: str            # "alliance", "horde", "neutral"
    min_buyout: int         # in copper
    market_value: int       # in copper (Nexus Hub market value)
    historical_value: int   # in copper (long-run historical)
    quantity: int
    num_auctions: int


@dataclass
class UndermineResponse:
    """Typed container for an Undermine Exchange API response."""

    source: str = "undermine"
    region: str = "us"
    realm_slug: str = ""
    faction: str = "neutral"
    endpoint: str = ""
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    records: list[UndermineAuctionRecord] = field(default_factory=list)
    is_fixture: bool = True


# ── Client ─────────────────────────────────────────────────────────────────────

class UndermineClient:
    """Stub client for the Undermine Exchange / Nexus Hub auction house API.

    Usage (fixture / stub mode — no API key required)::

        client = UndermineClient()
        response = client.get_fixture_response("area-52")

    Usage (real API — requires UNDERMINE_API_KEY in .env)::

        import os
        client = UndermineClient(api_key=os.environ["UNDERMINE_API_KEY"])
        response = client.fetch_realm_auctions("area-52", "neutral")

    Attributes:
        api_key: Optional API key from .env UNDERMINE_API_KEY.
        region: WoW region code ("us", "eu", "tw", "kr"). Default: "us".
    """

    BASE_URL: ClassVar[str] = "https://undermine.exchange/api"

    # Fixture auction data for stub / test mode.
    # Replace with real records once API access is configured.
    FIXTURE_RECORDS: ClassVar[list[dict]] = [
        {
            "item_id": 191528,
            "name": "Phial of Tepid Versatility",
            "min_buyout":       1_500_000,   # 150 gold
            "market_value":     1_600_000,   # 160 gold
            "historical_value": 1_450_000,   # 145 gold
            "quantity": 128,
            "num_auctions": 42,
        },
        {
            "item_id": 204783,
            "name": "Ironclaw Ore",
            "min_buyout":         80_000,    # 8 gold
            "market_value":       85_000,    # 8.5 gold
            "historical_value":   78_000,    # 7.8 gold
            "quantity": 4500,
            "num_auctions": 210,
        },
        {
            "item_id": 206448,
            "name": "Gleaming Obsidian Shard",
            "min_buyout":       2_000_000,   # 200 gold
            "market_value":     2_100_000,
            "historical_value": 1_950_000,
            "quantity": 35,
            "num_auctions": 12,
        },
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        region: str = "us",
    ) -> None:
        """Initialise the Undermine Exchange client.

        Args:
            api_key: API key from UNDERMINE_API_KEY env var. ``None`` → stub mode.
            region: WoW region ("us", "eu", "tw", "kr"). Default: "us".
        """
        self.api_key = api_key
        self.region = region

    # ── Real API methods (not yet implemented) ─────────────────────────────────

    def fetch_realm_auctions(
        self,
        realm_slug: str,
        faction: str = "neutral",
    ) -> UndermineResponse:
        """Fetch current AH auction data for a connected realm.

        TODO: Implement when API access is confirmed.
          1. Add ``httpx>=0.27`` to pyproject.toml.
          2. GET {BASE_URL}/v2/{region}/{realm_slug}/{faction}
          3. Parse response into UndermineAuctionRecord list.

        Args:
            realm_slug: Connected realm slug (e.g. "area-52").
            faction: "alliance", "horde", or "neutral".

        Returns:
            UndermineResponse with auction records.

        Raises:
            NotImplementedError: Until API credentials and HTTP are configured.
        """
        # TODO: replace with real HTTP call:
        # import httpx
        # resp = httpx.get(
        #     f"{self.BASE_URL}/v2/{self.region}/{realm_slug}/{faction}",
        #     headers={"Authorization": f"Bearer {self.api_key}"},
        #     timeout=30,
        # )
        # resp.raise_for_status()
        # return self._parse_response(resp.json(), realm_slug, faction)
        raise NotImplementedError(
            "Undermine API not yet configured. "
            "Set UNDERMINE_API_KEY in .env and implement HTTP call."
        )

    def fetch_market_values(self, realm_slug: str) -> UndermineResponse:
        """Fetch Nexus Hub market value estimates for all items on a realm.

        Args:
            realm_slug: Connected realm slug.

        Raises:
            NotImplementedError: Until API access is configured.
        """
        raise NotImplementedError(
            "Undermine market values fetch not yet implemented. "
            "See fetch_realm_auctions() for implementation pattern."
        )

    # ── Fixture / stub mode ────────────────────────────────────────────────────

    def get_fixture_response(
        self,
        realm_slug: str,
        faction: str = "neutral",
    ) -> UndermineResponse:
        """Return fixture auction data for testing and stub-mode ingestion.

        Use when UNDERMINE_API_KEY is not set. Records are synthetic but
        structurally valid — suitable for snapshot persistence and pipeline testing.

        Args:
            realm_slug: Realm slug to tag fixture records with.
            faction: Faction to tag fixture records with.

        Returns:
            UndermineResponse with sample auction records (is_fixture=True).
        """
        records = [
            UndermineAuctionRecord(
                item_id=r["item_id"],
                realm_slug=realm_slug,
                faction=faction,
                min_buyout=r["min_buyout"],
                market_value=r["market_value"],
                historical_value=r["historical_value"],
                quantity=r["quantity"],
                num_auctions=r["num_auctions"],
            )
            for r in self.FIXTURE_RECORDS
        ]
        logger.debug(
            "UndermineClient: returning %d fixture records for %s/%s",
            len(records), realm_slug, faction,
        )
        return UndermineResponse(
            region=self.region,
            realm_slug=realm_slug,
            faction=faction,
            endpoint=f"fixture/realm_auctions/{realm_slug}/{faction}",
            fetched_at=datetime.now(timezone.utc),
            records=records,
            is_fixture=True,
        )
