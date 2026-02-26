# Source Governance and Compliance Guardrails

## What this module does (and does not do)

The governance module (`wow_forecaster/governance/`) makes source usage policies
**explicit, configurable, and enforceable in code**.

### What it does

- Declares every data source used by the pipeline in one place (`config/sources.toml`).
- Stores technical constraints per source: rate limits, backoff policy, TTL/freshness
  thresholds, authentication requirements, provenance rules, and retention notes.
- Enforces enabled/disabled status and cooldown windows before any provider call is made.
- Classifies data freshness (`FRESH` / `AGING` / `STALE` / `CRITICAL` / `UNKNOWN`) based
  on per-source TTL thresholds — not a single hard-coded global constant.
- Provides three CLI commands for operational visibility.

### What it does NOT do

- **Interpret legal terms.** The `policy_notes` fields in `sources.toml` are
  researcher-authored informational reminders. They are not legal determinations.
- **Replace your own research.** Before using any API or data service, you are
  responsible for reviewing and complying with that service's terms. The pipeline
  cannot do this for you.
- **Grant or restrict rights.** Policy config is a technical control layer, not a
  legal one.

---

## Quick start

```bash
# List all registered sources
wow-forecaster list-sources

# Show full detail (rate limits, backoff, policy notes)
wow-forecaster list-sources --verbose

# Validate sources.toml is well-formed
wow-forecaster validate-source-policies

# Check freshness of all sources against their TTL policy
wow-forecaster check-source-freshness

# Check freshness for one realm and export a JSON report
wow-forecaster check-source-freshness --realm area-52 --export data/outputs/governance/
```

---

## Configuration: `config/sources.toml`

Each data source has a top-level `[sources.<id>]` block with the following
sub-tables.  All fields are documented below.

### Top-level fields

| Field | Type | Required | Description |
|---|---|---|---|
| `source_id` | string | yes | Unique identifier used throughout the pipeline |
| `display_name` | string | yes | Human-readable label for reports |
| `source_type` | string | yes | `auction_data`, `news_event`, `manual_event`, or `other` |
| `access_method` | string | yes | `api`, `export`, or `manual` |
| `requires_auth` | bool | yes | True if credentials must be configured |
| `enabled` | bool | yes | Set `false` until credentials are in `.env` |

### `[sources.<id>.rate_limit]`

Controls how frequently the pipeline calls this source.

| Field | Type | Default | Description |
|---|---|---|---|
| `requests_per_minute` | int | 0 | Max calls per minute. 0 = unlimited |
| `requests_per_hour` | int | 0 | Max calls per hour. 0 = unlimited |
| `burst_limit` | int | 0 | Max burst calls. 0 = unlimited |
| `cooldown_seconds` | float | 0.0 | Minimum seconds between consecutive calls |

**Usage in the pipeline:** `cooldown_seconds` is enforced by `run_preflight_checks()`
when `last_call_at` is supplied. The higher-level limits (`requests_per_minute`,
`requests_per_hour`) are metadata for future rate-limiting middleware; they are not
auto-enforced at the moment.

### `[sources.<id>.backoff]`

Retry / exponential backoff configuration for when a call fails.

| Field | Type | Default | Description |
|---|---|---|---|
| `strategy` | string | `"exponential"` | `exponential`, `linear`, or `fixed` |
| `base_seconds` | float | 1.0 | Delay for the first retry |
| `max_seconds` | float | 300.0 | Upper cap on retry delay |
| `jitter` | bool | true | ±25% random jitter to spread retries |
| `max_retries` | int | 5 | Give up after this many retries |

**Usage in the pipeline:** These values are metadata for the future real HTTP
client implementation. The fields are available via `SourcePolicy.backoff` so the
client can call `compute_backoff_delay(attempt, policy.backoff)` when needed.

### `[sources.<id>.freshness]`

Thresholds used by `check_source_freshness()` to classify data age.

| Field | Type | Description |
|---|---|---|
| `ttl_hours` | float | Data is `FRESH` below this age |
| `refresh_cadence_hours` | float | Intended call frequency (informational) |
| `stale_threshold_hours` | float | Beyond this: `STALE` |
| `critical_threshold_hours` | float | Beyond this: `CRITICAL` (forecasts unreliable) |

**Constraints:** `ttl_hours <= stale_threshold_hours <= critical_threshold_hours`
(validated by Pydantic on load).

**How age is calculated:** `check_source_freshness()` queries `ingestion_snapshots`
for `MAX(fetched_at) WHERE source = <id> AND success = 1`. Age is then
`utcnow() - last_fetched_at` in hours.

**Manual sources** (`requires_snapshot = false`) always return `UNKNOWN` because
they write no `ingestion_snapshots` records.

### `[sources.<id>.provenance]`

Documents what provenance metadata each call must record.

| Field | Type | Description |
|---|---|---|
| `requires_snapshot` | bool | Every call must write a JSON snapshot file to disk |
| `snapshot_format` | string | `"json"`, `"csv"`, or `"csv_or_json"` |
| `content_hash_required` | bool | Snapshot must include sha256 of response body |

**Usage in the pipeline:** IngestStage (once fully implemented) will consult
`requires_snapshot` to decide whether to call `save_snapshot()`. The `content_hash`
field in `ingestion_snapshots` holds the sha256.

### `[sources.<id>.retention]`

| Field | Type | Description |
|---|---|---|
| `raw_snapshot_days` | int | Days to keep raw snapshot files. 0 = indefinite |
| `notes` | string | Free-text explanation of retention rationale |

**Usage in the pipeline:** No automated cleanup is implemented yet. These fields
are the documented intent; a future `prune-snapshots` command will use them.

### `[sources.<id>.policy_notes]`

**These notes are informational reminders only. They are not legal advice.**

| Field | Type | Description |
|---|---|---|
| `access_type` | string | `authorized_api`, `export`, or `manual` |
| `requires_registered_account` | bool | Developer account needed? |
| `personal_research_only` | bool | Reminder that this system is for personal use |
| `notes` | string | Free-text notes for this source |

---

## Freshness status levels

| Status | Condition | Meaning |
|---|---|---|
| `FRESH` | `age < ttl_hours` | Data is current |
| `AGING` | `ttl_hours <= age < stale_threshold_hours` | Past TTL, still usable |
| `STALE` | `stale_threshold_hours <= age < critical_threshold_hours` | Stale, consider refresh |
| `CRITICAL` | `age >= critical_threshold_hours` | Do not use for forecasting |
| `UNKNOWN` | No snapshot in DB, or `requires_snapshot=false` | Cannot determine |

The `AGING` level is intentional — it gives you a warning before the data
reaches `STALE`, allowing proactive refresh.

**Why source-specific thresholds matter:** Auction data from `blizzard_api` becomes
stale after 3 hours (auctions expire every few hours). Manual event CSV data is
still valid after 14 days (patch cycles are ~6 weeks). A single global threshold
would either be too aggressive for event data or too lenient for auction data.

---

## How orchestration hooks enforce source policies

`HourlyOrchestrator._run_ingest()` in `pipeline/orchestrator.py` calls
`_run_governance_preflight(realm_slug)` before running `IngestStage`.

```
HourlyOrchestrator.run()
  └── _run_ingest(realm_slug)
        └── _run_governance_preflight(realm_slug)   ← NEW in v0.8.0
              ├── get_source_policy("undermine_exchange")
              ├── run_preflight_checks(policy)
              │     ├── Check: policy_present → always True
              │     ├── Check: enabled        → False if source is disabled
              │     └── Check: cooldown       → False if called too soon
              └── get_source_policy("blizzard_api")
                    └── same checks
```

**If a source is disabled** (the default for API sources until credentials are
configured), the check fails. Behaviour depends on `block_disabled_sources`:

- `block_disabled_sources = true` (default): The preflight call is logged as
  `WARNING` and the ingest call is silently skipped. The realm result is marked
  as `success=False` with a descriptive error message. **This is the safe default.**
  The pipeline continues for other realms and other pipeline stages.

- `block_disabled_sources = false`: A hard error is returned immediately, aborting
  the realm's ingest. Use this in integration tests where an enabled source is
  expected.

**In fixture mode** (the current state — all API sources are `enabled=false`): all
API preflight checks fail and are logged as warnings, but the `IngestStage` proceeds
with synthetic fixture data. This is intentional — the fixture data path does not
make real HTTP calls so governance blocking does not affect development workflows.

**When real HTTP is enabled**: set `enabled = true` in `sources.toml` and configure
credentials in `.env`. The preflight gate then acts as a last-resort block against
calling a source that was accidentally disabled or is mid-credential-rotation.

---

## Adding a new source

1. Add a `[sources.<new_id>]` block to `config/sources.toml`.
2. Fill in all required fields (see schema above).
3. Run `wow-forecaster validate-source-policies` to verify the config is valid.
4. The new source is immediately visible in `wow-forecaster list-sources`.
5. If the source needs a new ingestion client, add it to `wow_forecaster/ingestion/`
   and reference the source_id in `IngestStage`.

---

## Enabling a real API source

The API sources (`blizzard_api`, `undermine_exchange`) are `enabled = false` by
default. To activate one:

1. Obtain your credentials (see `.env.example`).
2. Add them to `.env`:
   ```env
   BLIZZARD_CLIENT_ID=your_client_id
   BLIZZARD_CLIENT_SECRET=your_client_secret
   UNDERMINE_API_KEY=your_key
   ```
3. In `config/sources.toml` (or `config/local.toml` for local-only overrides):
   ```toml
   [sources.blizzard_api]
   enabled = true
   ```
4. Run `wow-forecaster validate-source-policies` to confirm the config is valid.
5. Add `httpx>=0.27` to `pyproject.toml` and implement real HTTP in the client stubs.

---

## Verifying policy enforcement is working

### 1. Confirm sources.toml loads without error

```bash
wow-forecaster validate-source-policies
# Expected: "All source policies are valid."
```

### 2. List sources and confirm enabled/disabled status

```bash
wow-forecaster list-sources
# blizzard_api and undermine_exchange show [disabled]
# blizzard_news_manual and manual_event_csv show [ENABLED]
```

### 3. Check freshness (fresh DB — all UNKNOWN for API sources)

```bash
wow-forecaster check-source-freshness
# API sources: [UNKNOWN] (no ingestion_snapshots yet)
# Manual sources: [UNKNOWN] (requires_snapshot=false)
```

### 4. Run the hourly pipeline and check orchestrator warnings

```bash
wow-forecaster run-hourly-refresh --realm area-52
# Logs should show: "Governance[area-52][undermine_exchange]: blocked — Source is disabled"
# The realm result is still recorded; IngestStage runs in fixture mode
```

### 5. After ingest runs, check freshness again

```bash
wow-forecaster check-source-freshness
# API sources: still [UNKNOWN] (fixture mode writes ingestion_snapshots with source="undermine")
# If source IDs match, you should see [FRESH]
```

---

## Module file reference

| File | Purpose |
|---|---|
| `config/sources.toml` | Declarative source policy registry |
| `wow_forecaster/governance/__init__.py` | Public re-exports |
| `wow_forecaster/governance/models.py` | Pydantic models (SourcePolicy and sub-models) |
| `wow_forecaster/governance/registry.py` | Load, cache, look up, list source policies |
| `wow_forecaster/governance/freshness.py` | FreshnessResult, check_source_freshness() |
| `wow_forecaster/governance/preflight.py` | PreflightCheckResult, run_preflight_checks() |
| `wow_forecaster/governance/reporter.py` | ASCII formatters + JSON export |
| `wow_forecaster/config.py` | GovernanceConfig added to AppConfig |
| `config/default.toml` | [governance] section with defaults |
| `wow_forecaster/pipeline/orchestrator.py` | _run_governance_preflight() hook wired in |
| `wow_forecaster/cli.py` | list-sources, validate-source-policies, check-source-freshness |
| `tests/test_governance/` | 4 test files (models, freshness, preflight, registry) |

---

## Next steps

Once real HTTP clients are implemented (see "What's NOT Implemented Yet" in MEMORY.md):

1. **Wire backoff config into HTTP clients**: Use `policy.backoff.strategy`,
   `base_seconds`, `max_seconds`, and `jitter` in retry loops.
2. **Track `last_call_at` per source**: Store in a lightweight table or in-memory
   dict, and pass it to `run_preflight_checks()` so cooldown is actually enforced.
3. **Wire freshness check into orchestrator pre-flight**: After `_ensure_schema()`,
   call `check_all_sources_freshness()` for each realm and log warnings when API
   sources are `STALE` or `CRITICAL`.
4. **Implement `prune-snapshots` CLI command**: Use `retention.raw_snapshot_days`
   to delete old snapshot files from `data/raw/snapshots/`.
