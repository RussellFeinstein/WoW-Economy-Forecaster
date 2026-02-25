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

    # ── Real API methods (not yet implemented) ─────────────────────────────────

    def fetch_connected_realm_auctions(self, realm_slug: str) -> BlizzardAuctionResponse:
        """Fetch AH listings for a connected realm.

        TODO: Implement when BLIZZARD_CLIENT_ID + SECRET are set.
          1. Add ``httpx>=0.27`` to pyproject.toml.
          2. Call _ensure_token() for OAuth2 client credentials flow.
          3. realm_id = self.REALM_IDS.get(realm_slug) or fetch from API.
          4. GET /data/wow/connected-realm/{realm_id}/auctions?namespace=dynamic-{region}

        Args:
            realm_slug: Connected realm slug (e.g. "area-52").

        Raises:
            NotImplementedError: Until OAuth2 and HTTP are implemented.
        """
        # TODO: implement real call:
        # self._ensure_token()
        # realm_id = self.REALM_IDS.get(realm_slug) or self._lookup_realm_id(realm_slug)
        # base = self.BASE_URL_TEMPLATE.format(region=self.region)
        # import httpx
        # resp = httpx.get(
        #     f"{base}/data/wow/connected-realm/{realm_id}/auctions",
        #     params={"namespace": f"dynamic-{self.region}", "locale": "en_US"},
        #     headers={"Authorization": f"Bearer {self._access_token}"},
        #     timeout=60,
        # )
        # resp.raise_for_status()
        # return self._parse_response(resp.json(), realm_slug, realm_id)
        raise NotImplementedError(
            "Blizzard API not yet configured. "
            "Set BLIZZARD_CLIENT_ID and BLIZZARD_CLIENT_SECRET in .env."
        )

    def fetch_commodities(self) -> BlizzardAuctionResponse:
        """Fetch region-wide commodity AH data.

        Raises:
            NotImplementedError: Until OAuth2 and HTTP are implemented.
        """
        raise NotImplementedError(
            "Blizzard commodities fetch not yet implemented. "
            "Set BLIZZARD_CLIENT_ID and BLIZZARD_CLIENT_SECRET in .env."
        )

    # ── Token management (stub) ────────────────────────────────────────────────

    def _ensure_token(self) -> None:
        """Obtain or refresh the OAuth2 access token.

        TODO: Implement with real HTTP call:
          POST https://{region}.battle.net/oauth/token
          Body: grant_type=client_credentials
          Auth: Basic base64(client_id:client_secret)
        """
        raise NotImplementedError("OAuth2 token flow not yet implemented.")

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
