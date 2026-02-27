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

    # ── Real methods ───────────────────────────────────────────────────────────

    def fetch_recent_news(self, limit: int = 20) -> NewsResponse:
        """Fetch recent WoW news articles via the official RSS feed.

        Parses ``https://news.blizzard.com/en-us/rss/world-of-warcraft`` using
        feedparser and classifies each entry by title keywords.

        Args:
            limit: Maximum number of articles to return (most recent first).

        Returns:
            NewsResponse with real articles (is_fixture=False).

        Raises:
            Exception: If the RSS feed is unreachable or malformed.
        """
        import re
        import time

        import feedparser

        feed = feedparser.parse(self.RSS_URL)
        items: list[NewsItem] = []

        for entry in feed.entries[:limit]:
            # Publication timestamp
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                ts = time.mktime(entry.published_parsed)
                published_at = datetime.fromtimestamp(ts, tz=timezone.utc)
            else:
                published_at = datetime.now(timezone.utc)

            title   = getattr(entry, "title", "")
            summary = getattr(entry, "summary", "")[:500]
            link    = getattr(entry, "link", "")

            title_lower = title.lower()

            # Detect patch notes
            is_patch_notes = any(
                kw in title_lower
                for kw in ("hotfix", "patch notes", "ptr development", "maintenance", "update notes")
            )

            # Extract patch version like "11.2.7" or "11.1" from title
            patch_version: str | None = None
            m = re.search(r"\b(\d+\.\d+(?:\.\d+)?)\b", title)
            if m and is_patch_notes:
                patch_version = m.group(1)

            # Classify category
            if "patch" in title_lower or "hotfix" in title_lower or "ptr" in title_lower:
                category = "patch-notes"
            elif any(kw in title_lower for kw in ("announce", "reveal", "launch", "goes live")):
                category = "announcement"
            elif any(kw in title_lower for kw in ("developer", "dev", "blog", "inside")):
                category = "developer-update"
            else:
                category = "announcement"

            items.append(
                NewsItem(
                    title=title,
                    url=link,
                    published_at=published_at,
                    category=category,
                    summary=summary,
                    is_patch_notes=is_patch_notes,
                    patch_version=patch_version,
                )
            )

        return NewsResponse(
            source="blizzard_news",
            endpoint=self.RSS_URL,
            fetched_at=datetime.now(timezone.utc),
            items=items,
            is_fixture=False,
        )

    def fetch_patch_notes(self) -> NewsResponse:
        """Fetch only patch note articles from the RSS feed.

        Returns:
            NewsResponse filtered to patch-notes category.
        """
        full = self.fetch_recent_news(limit=50)
        patch_items = [i for i in full.items if i.is_patch_notes]
        return NewsResponse(
            source=full.source,
            endpoint=full.endpoint,
            fetched_at=full.fetched_at,
            items=patch_items,
            is_fixture=False,
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
