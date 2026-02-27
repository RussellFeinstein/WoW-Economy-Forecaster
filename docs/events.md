# Events System

This document describes the canonical event calendar, how to add events, and how
event features are generated without look-ahead bias.

---

## Overview

The events system connects real-world WoW game events (patches, seasons, RTWF, holidays)
to the market time series so the ML model can learn demand patterns around them.

**Pipeline position:**

```
build-events  ->  build-datasets  ->  train-model  ->  run-daily-forecast
```

`build-events` must run (at least once) before `build-datasets` so that all 8 event
feature columns are populated.

---

## Seed files

### `config/events/tww_events.json`

The canonical event calendar for The War Within (TWW, patch 11.0–11.1).

Each entry is a JSON object with the following fields:

| Field | Type | Required | Description |
|---|---|---|---|
| `slug` | string | yes | Stable unique identifier (e.g. `tww-rtwf-nerubar-s1`). Never change after initial seed. |
| `display_name` | string | yes | Human-readable name shown in reports. |
| `event_type` | string | yes | One of 26 `EventType` values (see `event_taxonomy.py`). |
| `scope` | string | yes | `global`, `region`, `realm_cluster`, or `faction`. |
| `severity` | string | yes | `negligible`, `minor`, `moderate`, `major`, or `critical`. |
| `expansion_slug` | string | yes | `tww`, `midnight`, etc. |
| `start_date` | ISO date | yes | First day of the event (`YYYY-MM-DD`). |
| `end_date` | ISO date | no | Last day of the event. Null = instantaneous (single day). |
| `announced_at` | ISO datetime | **critical** | When the event was publicly announced. Events with `null` here are **excluded from all feature computation** (Layer 1 leakage guard). |
| `is_recurring` | bool | no | True for repeating events (DMF, holidays). |
| `recurrence_rule` | string | no | iCal RRULE format (e.g. `FREQ=MONTHLY;BYDAY=1SU`). |
| `patch_version` | string | no | Patch version associated with the event. |
| `notes` | string | no | Free-text research notes (not used by ML). |

**Key invariant:** `end_date >= start_date`. Violation causes `build-events` to fail.

#### How to add a new event

1. Open `config/events/tww_events.json`.
2. Append a new JSON object following the schema above.
3. Set `announced_at` to the UTC timestamp of the official Blizzard announcement.
   - For well-known recurring events (holidays, DMF), use `"2004-11-23T00:00:00+00:00"` (WoW launch).
   - For content droughts (no formal announcement), use the `start_date` as the `announced_at`.
4. Run `wow-forecaster build-events` to upsert and regenerate Parquet.
5. Run `wow-forecaster build-datasets` to propagate the new event into features.

---

### `config/events/tww_event_impacts.json`

Category-level economic impact annotations for each event.

Each entry maps one `(event_slug, archetype_category)` pair to an expected market impact:

| Field | Type | Required | Description |
|---|---|---|---|
| `event_slug` | string | yes | Must match a `slug` in `tww_events.json`. |
| `archetype_category` | string | yes | One of the `ArchetypeCategory` values: `consumable`, `mat`, `gear`, `enchant`, `gem`, `prof_tool`, `reagent`, `trade_good`. |
| `impact_direction` | string | yes | `spike`, `crash`, `mixed`, or `neutral`. |
| `typical_magnitude` | float | no | Expected fractional price change (e.g. `0.40` = +40%; `-0.30` = -30%). |
| `lag_days` | int | no | Days after event start when impact begins. Negative = pre-event run-up. |
| `duration_days` | int | no | How many days the impact lasts. Null = persists until event end. |
| `source` | string | no | Provenance string (default `"seed"`). |
| `notes` | string | no | Research rationale. |

**Validation rules:**
- `impact_direction` must be one of `spike`, `crash`, `mixed`, `neutral`.
- `archetype_category` must be a valid `ArchetypeCategory` enum value.
- Each `(event_slug, archetype_category)` pair must be unique.
- `event_slug` must reference a slug present in the events file.

JSON entries that have only a `"_comment"` key (no `event_slug`) are treated as
comments and silently ignored.

#### How to add an impact annotation

1. Open `config/events/tww_event_impacts.json`.
2. Append a new JSON object with at minimum `event_slug`, `archetype_category`,
   and `impact_direction`.
3. Set `typical_magnitude` as a fraction of the baseline price (positive = spike).
4. Run `wow-forecaster build-events` to upsert and regenerate Parquet.

---

## Database tables

| Table | Description |
|---|---|
| `wow_events` | Canonical event calendar (one row per event occurrence). |
| `event_category_impacts` | Category-level impact seeds; `(event_id, archetype_category)` unique. |
| `event_archetype_impacts` | Specific archetype-level impacts (FK to `economic_archetypes`); used for per-archetype direction override. |

---

## Event features

All 8 event columns are computed in `wow_forecaster/features/event_features.py`
and added to every row in the training and inference Parquet files.

### Feature definitions

| Column | Type | Null? | Description |
|---|---|---|---|
| `event_active` | bool | never | True if any known event is active on `obs_date`. |
| `event_days_to_next` | float | yes | Days until next known future event. Null if none. |
| `event_days_since_last` | float | yes | Days since last completed event ended. Null if none. |
| `event_severity_max` | string | yes | Max severity of currently-active events. Null if none active. |
| `event_archetype_impact` | string | yes | Impact direction from `event_archetype_impacts` for this archetype. Null if no specific record. |
| `event_impact_magnitude` | float | yes | Expected price change fraction from `event_category_impacts` for this archetype's category. Null if no record. |
| `days_until_major_event` | float | yes | Days until next known MAJOR or CRITICAL event. Leakage-safe. |
| `is_pre_event_window` | bool | never | True if `days_until_major_event` is 1–7. False otherwise. |

### Leakage prevention (3 layers)

**Layer 1 — DB filter** (`load_known_events()`):
Only events with `announced_at IS NOT NULL` are loaded. Events without an
announcement date are never used in feature computation.

**Layer 2 — Per-row guard** (`compute_event_features()`):
For each row at date `D`, we compute `as_of = D 23:59:59 UTC`. The
`WoWEvent.is_known_at(as_of)` method returns `True` only when
`announced_at <= as_of`. This ensures:
- An event announced at 17:00 UTC on the same calendar day as the observation
  *is* included (same-day announcement is known by end-of-day).
- An event announced the day *after* the observation is *excluded*.
- `days_until_major_event` and `is_pre_event_window` only reflect events that
  were publicly known on `obs_date` — no future information leaks in.

**Layer 3 — Quality heuristic** (`build_quality_report()`):
Checks that `event_days_to_next >= 0` for all rows, catching any logic error
that would treat a past event as upcoming.

---

## Commands

### Initialize and seed (first time)

```bash
wow-forecaster init-db
wow-forecaster build-events
```

### Regenerate after adding events or impacts

```bash
wow-forecaster build-events
wow-forecaster build-datasets     # Rebuild feature Parquet with updated events
```

### Full new-developer setup

```bash
wow-forecaster init-db
wow-forecaster import-events config/events/tww_events.json  # optional (build-events handles this)
wow-forecaster build-events
wow-forecaster build-datasets
wow-forecaster train-model
wow-forecaster run-daily-forecast
```

### Override seed file paths

```bash
wow-forecaster build-events \
  --events-file  path/to/custom_events.json \
  --impacts-file path/to/custom_impacts.json \
  --output-dir   data/processed/events
```

---

## Parquet outputs

`build-events` writes two Parquet files to `data/processed/events/`:

### `events.parquet`

| Column | Type | Description |
|---|---|---|
| `event_id` | string | Stable slug (e.g. `tww-rtwf-nerubar-s1`). |
| `event_name` | string | Display name. |
| `event_type` | string | EventType enum value. |
| `scope` | string | EventScope enum value. |
| `severity` | string | EventSeverity enum value. |
| `expansion_slug` | string | Expansion this event belongs to. |
| `start_ts` | date32 | Event start date. |
| `end_ts` | date32 | Event end date (null for instantaneous). |
| `announced_ts` | date32 | Date when event was announced (date part only). |
| `source` | string | Always `"seed"` for seeded data. |
| `metadata` | string | JSON blob: patch_version, recurrence_rule, notes, is_recurring. |

### `event_category_impacts.parquet`

| Column | Type | Description |
|---|---|---|
| `event_id` | string | Event slug. |
| `archetype_category` | string | ArchetypeCategory enum value. |
| `impact_direction` | string | spike / crash / mixed / neutral. |
| `typical_magnitude` | float32 | Expected fractional price change. |
| `lag_days` | int32 | Days after event start when impact begins. |
| `duration_days` | int32 | Duration of impact in days (null = until event end). |
| `source` | string | Provenance (default `"seed"`). |
| `notes` | string | Research notes. |

---

## Coverage: TWW events in seed (v0.9.0)

| Slug | Type | Severity | Announced | Impacts |
|---|---|---|---|---|
| `tww-launch` | expansion_launch | critical | 2023-11-03 | consumable, gear, mat, enchant, gem, reagent |
| `tww-prepatch-110` | expansion_prepatch | major | 2024-07-10 | consumable, mat, gear |
| `tww-rtwf-nerubar-s1` | rtwf | major | 2024-08-19 | consumable, mat, enchant, gem, gear, reagent |
| `tww-s1-start` | season_start | major | 2024-08-19 | consumable, enchant, gem, gear, mat, reagent |
| `tww-patch-1105` | minor_patch | minor | 2024-10-10 | consumable, gear, mat |
| `tww-s2-start` | season_start | major | 2025-02-11 | consumable, enchant, gem, gear, mat, reagent |
| `tww-patch-111` | major_patch | major | 2025-01-21 | consumable, enchant, gem, gear, mat, reagent |
| `tww-rtwf-liberation-s2` | rtwf | major | 2025-02-28 | consumable, mat, enchant, gem, gear, reagent |
| `tww-s1-end` | season_end | moderate | 2025-02-11 | consumable, gear, mat, enchant, gem |
| `tww-content-drought-s1-end` | content_drought | moderate | 2025-01-15 | consumable, mat, gear |
| `tww-blizzcon-2024` | blizzcon | minor | 2024-08-01 | mat, collection |
| `tww-darkmoon-faire-recurring` | holiday_event | minor | 2004-11-23 | reagent, consumable |
| `tww-winter-veil-2024` | holiday_event | minor | 2004-11-23 | consumable |
| `tww-brewfest-2024` | holiday_event | minor | 2004-11-23 | mat |
| `tww-trading-post-2024-09` | trading_post_reset | negligible | 2024-08-20 | (none) |

---

## Adding events for Midnight expansion

When building Midnight transfer learning datasets, add Midnight events using the same
JSON structure with `expansion_slug: "midnight"`. The `announced_at` field remains
critical for leakage prevention — set it to the BlizzCon reveal date for expansion
events, and to the official patch announcement for content patches.
