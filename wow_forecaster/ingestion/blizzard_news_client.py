"""
Blizzard WoW news and patch notes client — typed stub.

Source:  https://news.blizzard.com/en-us/world-of-warcraft
RSS:     https://news.blizzard.com/en-us/rss/world-of-warcraft

No API key required — public news feed.

Purpose:
  Detect patch announcements, content updates, and economic event windows
  before they happen by monitoring official Blizzard communications.
  News items feed the event-signal pipeline for WoWEvent auto-detection.

When implementing:
  1. Add ``httpx>=0.27`` and ``feedparser>=6.0`` to pyproject.toml.
  2. Implement fetch_recent_news() by parsing the RSS feed.
  3. Implement extract_wow_events() to map news items → WoWEvent candidates.
  4. Run as a separate ingestion pass (less frequent than AH data — daily is fine).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import ClassVar, Optional

logger = logging.getLogger(__name__)


# ── Response types ─────────────────────────────────────────────────────────────

@dataclass
class NewsItem:
    """A single WoW news article or patch note."""

    title: str
    url: str
    published_at: datetime
    category: str           # "patch-notes" | "announcement" | "developer-update" | "hotfixes"
    summary: str = ""
    is_patch_notes: bool = False
    patch_version: Optional[str] = None


@dataclass
class NewsResponse:
    """Typed container for a Blizzard news fetch."""

    source: str = "blizzard_news"
    endpoint: str = ""
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    items: list[NewsItem] = field(default_factory=list)
    is_fixture: bool = True


# ── Client ─────────────────────────────────────────────────────────────────────

class BlizzardNewsClient:
    """Stub client for ingesting Blizzard WoW news and patch announcements.

    Usage (fixture / stub mode)::

        client = BlizzardNewsClient()
        response = client.get_fixture_response()

    Usage (real RSS — after adding httpx + feedparser)::

        client = BlizzardNewsClient()
        response = client.fetch_recent_news(limit=20)
    """

    NEWS_URL: ClassVar[str] = "https://news.blizzard.com/en-us/world-of-warcraft"
    RSS_URL: ClassVar[str] = "https://news.blizzard.com/en-us/rss/world-of-warcraft"

    FIXTURE_ITEMS: ClassVar[list[dict]] = [
        {
            "title": "The War Within — Patch 11.1.5 PTR Development Notes",
            "url": "https://news.blizzard.com/en-us/article/24200001",
            "published_at": "2025-12-10T18:00:00Z",
            "category": "patch-notes",
            "summary": "Patch 11.1.5 brings new seasonal content to The War Within.",
            "is_patch_notes": True,
            "patch_version": "11.1.5",
        },
        {
            "title": "World of Warcraft: Midnight — Announced at BlizzCon 2025",
            "url": "https://news.blizzard.com/en-us/article/24200002",
            "published_at": "2025-11-03T18:00:00Z",
            "category": "announcement",
            "summary": "The next chapter of World of Warcraft takes us to Quel'Thalas.",
            "is_patch_notes": False,
            "patch_version": None,
        },
        {
            "title": "Season of Discovery — Trading Post November 2025",
            "url": "https://news.blizzard.com/en-us/article/24200003",
            "published_at": "2025-11-01T17:00:00Z",
            "category": "developer-update",
            "summary": "New items available in the Trading Post for November.",
            "is_patch_notes": False,
            "patch_version": None,
        },
    ]

    # ── Real methods (not yet implemented) ────────────────────────────────────

    def fetch_recent_news(self, limit: int = 20) -> NewsResponse:
        """Fetch recent WoW news articles via RSS feed.

        TODO: Implement with feedparser:
          1. Add ``feedparser>=6.0`` to pyproject.toml.
          2. import feedparser; feed = feedparser.parse(self.RSS_URL)
          3. Map feed.entries → NewsItem list.

        Args:
            limit: Maximum number of articles to return.

        Raises:
            NotImplementedError: Until feedparser integration is implemented.
        """
        # TODO: implement:
        # import feedparser
        # feed = feedparser.parse(self.RSS_URL)
        # items = [self._entry_to_news_item(e) for e in feed.entries[:limit]]
        # return NewsResponse(endpoint=self.RSS_URL, fetched_at=utcnow(), items=items, is_fixture=False)
        raise NotImplementedError(
            "Blizzard news fetch not yet implemented. "
            "Add httpx + feedparser to pyproject.toml and implement RSS parsing."
        )

    def fetch_patch_notes(self) -> NewsResponse:
        """Fetch only patch note articles.

        Raises:
            NotImplementedError: Until fetch_recent_news() is implemented.
        """
        raise NotImplementedError(
            "Patch notes fetch not yet implemented. See fetch_recent_news()."
        )

    # ── Fixture / stub mode ────────────────────────────────────────────────────

    def get_fixture_response(self) -> NewsResponse:
        """Return fixture news data for testing and stub-mode ingestion.

        Returns:
            NewsResponse with sample items (is_fixture=True).
        """
        items = [
            NewsItem(
                title=d["title"],
                url=d["url"],
                published_at=datetime.fromisoformat(d["published_at"].replace("Z", "+00:00")),
                category=d["category"],
                summary=d["summary"],
                is_patch_notes=d["is_patch_notes"],
                patch_version=d["patch_version"],
            )
            for d in self.FIXTURE_ITEMS
        ]
        logger.debug("BlizzardNewsClient: returning %d fixture items", len(items))
        return NewsResponse(
            endpoint="fixture/recent-news",
            fetched_at=datetime.now(timezone.utc),
            items=items,
            is_fixture=True,
        )
