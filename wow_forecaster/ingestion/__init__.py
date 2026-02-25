"""
Ingestion layer — provider clients, snapshot persistence, and normalization contracts.

Submodules:
  undermine_client    — Undermine Exchange / Nexus Hub AH data (stub)
  blizzard_client     — Blizzard Game Data API AH data (stub)
  blizzard_news_client— Blizzard news feed for event signals (stub)
  snapshot            — Disk-based raw payload persistence
  event_csv           — CSV import parser for manual WoWEvent records

Credential placement (.env, gitignored):
  UNDERMINE_API_KEY          — Undermine Exchange API key
  BLIZZARD_CLIENT_ID         — Blizzard OAuth2 client ID
  BLIZZARD_CLIENT_SECRET     — Blizzard OAuth2 client secret
  BLIZZARD_REGION            — Region override (default: us)
"""
