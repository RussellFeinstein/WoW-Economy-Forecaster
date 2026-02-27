"""
Blizzard Game Data API client — typed stub with fixture data.

API:   https://us.api.blizzard.com/data/wow/
Docs:  https://develop.battle.net/documentation/world-of-warcraft/game-data-apis

Credential setup (.env, gitignored):
  BLIZZARD_CLIENT_ID=your_client_id
  BLIZZARD_CLIENT_SECRET=your_client_secret
  BLIZZARD_REGION=us          # us | eu | kr | tw  (default: us)

OAuth2 flow:
  Client credentials grant — no user interaction needed.
  POST https://us.battle.net/oauth/token
    → Body: grant_type=client_credentials
    → Auth: Basic (client_id:client_secret)
    → Returns: {"access_token": "...", "expires_in": 86399}

AH endpoints (after getting token):
  Connected realm auctions:
    GET /data/wow/connected-realm/{realmId}/auctions
    ?namespace=dynamic-us&locale=en_US
  Commodities (region-wide):
    GET /data/wow/auctions/commodities
    ?namespace=dynamic-us&locale=en_US

When implementing:
  1. Add ``httpx>=0.27`` to pyproject.toml.
  2. Implement _ensure_token() for OAuth2 client credentials.
  3. Fill in fetch_connected_realm_auctions() and fetch_commodities().
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import ClassVar, Optional

logger = logging.getLogger(__name__)


# ── Response types ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BlizzardAuctionRecord:
    """A single AH listing as returned by the Blizzard Game Data API."""

    item_id: int
    realm_id: int
    realm_slug: str
    buyout: int         # in copper (0 if no buyout — bid-only)
    bid: int            # minimum bid in copper
    unit_price: int     # for commodities: copper per unit; 0 for non-commodity
    quantity: int
    time_left: str      # "SHORT" | "MEDIUM" | "LONG" | "VERY_LONG"


@dataclass
class BlizzardAuctionResponse:
    """Typed container for a Blizzard AH API response."""

    source: str = "blizzard_api"
    region: str = "us"
    realm_id: int = 0
    realm_slug: str = ""
    endpoint: str = ""
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    records: list[BlizzardAuctionRecord] = field(default_factory=list)
    is_fixture: bool = True


# ── Client ─────────────────────────────────────────────────────────────────────

class BlizzardClient:
    """Stub client for the Blizzard Game Data API — auction house data.

    Usage (fixture / stub mode — no credentials required)::

        client = BlizzardClient()
        response = client.get_fixture_response("area-52")

    Usage (real API — requires BLIZZARD_CLIENT_ID + SECRET in .env)::

        import os
        client = BlizzardClient(
            client_id=os.environ["BLIZZARD_CLIENT_ID"],
            client_secret=os.environ["BLIZZARD_CLIENT_SECRET"],
        )
        response = client.fetch_connected_realm_auctions("area-52")

    Common US realm IDs are pre-mapped in REALM_IDS. Add more as needed.
    """

    BASE_URL_TEMPLATE: ClassVar[str] = "https://{region}.api.blizzard.com"
    TOKEN_URL_TEMPLATE: ClassVar[str] = "https://{region}.battle.net/oauth/token"

    # slug → Blizzard connected-realm ID (US region)
    # Full list: /data/wow/connected-realm/index?namespace=dynamic-us
    REALM_IDS: ClassVar[dict[str, int]] = {
        "area-52":   1175,
        "illidan":   1185,
        "stormrage": 1146,
        "tichondrius": 1190,
        "mal'ganis": 1184,
        "bleeding-hollow": 1168,
        "whisperwind": 1196,
    }

    FIXTURE_RECORDS: ClassVar[list[dict]] = [
        {
            "item_id": 191528,
            "buyout":    1_520_000,   # 152 gold
            "bid":       1_400_000,
            "unit_price": 0,
            "quantity": 1,
            "time_left": "VERY_LONG",
        },
        {
            "item_id": 204783,
            "buyout":      82_000,    # 8.2 gold
            "bid":         75_000,
            "unit_price":      0,
            "quantity": 200,
            "time_left": "LONG",
        },
        {
            "item_id": 206448,
            "buyout":   2_050_000,   # 205 gold
            "bid":      1_900_000,
            "unit_price": 0,
            "quantity": 1,
            "time_left": "MEDIUM",
        },
    ]

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        region: str = "us",
    ) -> None:
        """Initialise the Blizzard Game Data API client.

        Args:
            client_id: OAuth2 client ID from BLIZZARD_CLIENT_ID env var.
            client_secret: OAuth2 client secret from BLIZZARD_CLIENT_SECRET env var.
            region: WoW region ("us", "eu", "kr", "tw"). Default: "us".
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.region = region
        self._access_token: Optional[str] = None

    # ── Real API methods ───────────────────────────────────────────────────────

    def fetch_commodities(self) -> BlizzardAuctionResponse:
        """Fetch region-wide commodity AH data (one call covers all US realms).

        Commodities are region-wide since patch 9.2.7.  This is the primary
        data source for crafting materials, consumables, and tradeable goods.

        Endpoint::

            GET /data/wow/auctions/commodities
                ?namespace=dynamic-{region}&locale=en_US

        Returns:
            BlizzardAuctionResponse with commodity records (is_fixture=False).

        Raises:
            httpx.HTTPStatusError: On non-2xx API response.
            RuntimeError: If client has no credentials.
        """
        import httpx

        if not self.client_id or not self.client_secret:
            raise RuntimeError(
                "BLIZZARD_CLIENT_ID and BLIZZARD_CLIENT_SECRET must be set in .env."
            )
        self._ensure_token()
        base = self.BASE_URL_TEMPLATE.format(region=self.region)
        resp = httpx.get(
            f"{base}/data/wow/auctions/commodities",
            params={"namespace": f"dynamic-{self.region}", "locale": "en_US"},
            headers={"Authorization": f"Bearer {self._access_token}"},
            timeout=120.0,  # Commodity response can be 15-30 MB
        )
        resp.raise_for_status()
        return self._parse_commodities_response(resp.json())

    def fetch_connected_realm_auctions(self, realm_slug: str) -> BlizzardAuctionResponse:
        """Fetch AH listings for a single connected realm (non-commodity BoE gear).

        Note: For region-wide commodity data (crafting mats, consumables) use
        ``fetch_commodities()`` instead — it's a single call for all realms.

        Args:
            realm_slug: Connected realm slug (e.g. "area-52").

        Returns:
            BlizzardAuctionResponse with per-realm auction records.

        Raises:
            ValueError:            If realm_slug is not in REALM_IDS.
            httpx.HTTPStatusError: On non-2xx API response.
        """
        import httpx

        if not self.client_id or not self.client_secret:
            raise RuntimeError(
                "BLIZZARD_CLIENT_ID and BLIZZARD_CLIENT_SECRET must be set in .env."
            )
        realm_id = self.REALM_IDS.get(realm_slug)
        if realm_id is None:
            raise ValueError(
                f"Unknown realm slug '{realm_slug}'. "
                f"Add it to BlizzardClient.REALM_IDS or use fetch_commodities() "
                f"for region-wide commodity data."
            )
        self._ensure_token()
        base = self.BASE_URL_TEMPLATE.format(region=self.region)
        resp = httpx.get(
            f"{base}/data/wow/connected-realm/{realm_id}/auctions",
            params={"namespace": f"dynamic-{self.region}", "locale": "en_US"},
            headers={"Authorization": f"Bearer {self._access_token}"},
            timeout=120.0,
        )
        resp.raise_for_status()
        return self._parse_realm_response(resp.json(), realm_slug, realm_id)

    # ── Token management ───────────────────────────────────────────────────────

    def _ensure_token(self) -> None:
        """Obtain (or reuse cached) OAuth2 client-credentials access token.

        Token is cached in ``self._access_token`` for the lifetime of this
        client instance.  Blizzard tokens are valid for 24 hours, so one
        token per hourly pipeline run is safe.

        Raises:
            httpx.HTTPStatusError: If the token endpoint returns a non-2xx status.
        """
        if self._access_token is not None:
            return

        import base64

        import httpx

        token_url = self.TOKEN_URL_TEMPLATE.format(region=self.region)
        credentials = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()

        resp = httpx.post(
            token_url,
            headers={
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={"grant_type": "client_credentials"},
            timeout=30.0,
        )
        resp.raise_for_status()
        self._access_token = resp.json()["access_token"]
        logger.info("Blizzard OAuth2 token obtained for region=%s", self.region)

    # ── Response parsers ───────────────────────────────────────────────────────

    def _parse_commodities_response(self, data: dict) -> BlizzardAuctionResponse:
        """Parse the /auctions/commodities JSON response into typed records."""
        records = []
        for auction in data.get("auctions", []):
            item_id = auction.get("item", {}).get("id")
            if item_id is None:
                continue
            records.append(
                BlizzardAuctionRecord(
                    item_id=int(item_id),
                    realm_id=0,
                    realm_slug=self.region,  # "us" — region-wide
                    buyout=0,
                    bid=0,
                    unit_price=int(auction.get("unit_price", 0)),
                    quantity=int(auction.get("quantity", 0)),
                    time_left="VERY_LONG",  # Commodities have no expiry time_left
                )
            )
        return BlizzardAuctionResponse(
            region=self.region,
            realm_id=0,
            realm_slug=self.region,
            endpoint=f"data/wow/auctions/commodities",
            fetched_at=datetime.now(timezone.utc),
            records=records,
            is_fixture=False,
        )

    def _parse_realm_response(
        self, data: dict, realm_slug: str, realm_id: int
    ) -> BlizzardAuctionResponse:
        """Parse a connected-realm /auctions JSON response into typed records."""
        records = []
        for auction in data.get("auctions", []):
            item_id = auction.get("item", {}).get("id")
            if item_id is None:
                continue
            records.append(
                BlizzardAuctionRecord(
                    item_id=int(item_id),
                    realm_id=realm_id,
                    realm_slug=realm_slug,
                    buyout=int(auction.get("buyout", 0)),
                    bid=int(auction.get("bid", 0)),
                    unit_price=int(auction.get("unit_price", 0)),
                    quantity=int(auction.get("quantity", 1)),
                    time_left=auction.get("time_left", "VERY_LONG"),
                )
            )
        return BlizzardAuctionResponse(
            region=self.region,
            realm_id=realm_id,
            realm_slug=realm_slug,
            endpoint=f"data/wow/connected-realm/{realm_id}/auctions",
            fetched_at=datetime.now(timezone.utc),
            records=records,
            is_fixture=False,
        )

    # ── Fixture / stub mode ────────────────────────────────────────────────────

    def get_fixture_response(self, realm_slug: str) -> BlizzardAuctionResponse:
        """Return fixture AH data for testing and stub-mode ingestion.

        Args:
            realm_slug: Realm slug to tag fixture records with.

        Returns:
            BlizzardAuctionResponse with sample records (is_fixture=True).
        """
        realm_id = self.REALM_IDS.get(realm_slug, 9999)
        records = [
            BlizzardAuctionRecord(
                item_id=r["item_id"],
                realm_id=realm_id,
                realm_slug=realm_slug,
                buyout=r["buyout"],
                bid=r["bid"],
                unit_price=r["unit_price"],
                quantity=r["quantity"],
                time_left=r["time_left"],
            )
            for r in self.FIXTURE_RECORDS
        ]
        logger.debug(
            "BlizzardClient: returning %d fixture records for %s (realm_id=%d)",
            len(records), realm_slug, realm_id,
        )
        return BlizzardAuctionResponse(
            region=self.region,
            realm_id=realm_id,
            realm_slug=realm_slug,
            endpoint=f"fixture/connected-realm/{realm_slug}/auctions",
            fetched_at=datetime.now(timezone.utc),
            records=records,
            is_fixture=True,
        )
