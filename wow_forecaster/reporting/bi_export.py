"""
Business Intelligence export — star-schema CSV/Parquet for Power BI and Tableau.

Generates dimension and fact tables from the SQLite database that load
directly into BI tools without transformation. All tables are FK-linked
by integer IDs for efficient joins.

Usage::

    from wow_forecaster.reporting.bi_export import export_star_schema
    export_star_schema("wow_forecaster.db", "data/exports/bi", "us")
"""

from __future__ import annotations

import csv
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


def export_star_schema(
    db_path: str | Path,
    output_dir: str | Path,
    realm: str,
    include_backtest: bool = False,
    fmt: str = "csv",
) -> dict[str, Path]:
    """Generate a full star-schema export bundle.

    Args:
        db_path:          Path to the SQLite database.
        output_dir:       Directory to write export files.
        realm:            Realm slug to filter by.
        include_backtest: If True, include fact_backtest table.
        fmt:              "csv" or "parquet".

    Returns:
        Dict mapping table name to written file Path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    written: dict[str, Path] = {}

    # ── Dimension tables ─────────────────────────────────────────────────

    written["dim_archetypes"] = _export_table(
        conn, out, "dim_archetypes", fmt,
        """
        SELECT archetype_id, slug, display_name, category_tag,
               sub_tag, is_transferable, transfer_confidence
        FROM   economic_archetypes
        ORDER  BY archetype_id
        """,
    )

    written["dim_events"] = _export_table(
        conn, out, "dim_events", fmt,
        """
        SELECT event_id, slug, display_name, event_type,
               scope, severity, expansion_slug,
               start_date, end_date
        FROM   wow_events
        ORDER  BY start_date
        """,
    )

    written["dim_items"] = _export_table(
        conn, out, "dim_items", fmt,
        """
        SELECT i.item_id, i.name, i.category_id, i.archetype_id,
               i.expansion_slug, i.quality, i.is_crafted, i.is_boe,
               ea.category_tag, ea.display_name AS archetype_name
        FROM   items i
        LEFT JOIN economic_archetypes ea ON i.archetype_id = ea.archetype_id
        ORDER  BY i.item_id
        """,
    )

    # Construct dim_dates from the date range present in normalized observations
    written["dim_dates"] = _export_table(
        conn, out, "dim_dates", fmt,
        """
        WITH date_range AS (
            SELECT DISTINCT date(observed_at) AS date_val
            FROM   market_observations_normalized
            WHERE  realm_slug = ?
        )
        SELECT date_val                                     AS date,
               cast(strftime('%w', date_val) as integer)    AS day_of_week,
               cast(strftime('%j', date_val) as integer)    AS day_of_year,
               cast(strftime('%W', date_val) as integer)    AS week_of_year,
               cast(strftime('%m', date_val) as integer)    AS month,
               CASE WHEN cast(strftime('%w', date_val) as integer) IN (0, 6)
                    THEN 1 ELSE 0 END                       AS is_weekend
        FROM   date_range
        ORDER  BY date_val
        """,
        params=[realm],
    )

    # ── Fact tables ──────────────────────────────────────────────────────

    written["fact_prices"] = _export_table(
        conn, out, "fact_prices", fmt,
        """
        SELECT i.archetype_id,
               n.realm_slug,
               date(n.observed_at)       AS obs_date,
               AVG(n.price_gold)         AS avg_price_gold,
               MIN(n.price_gold)         AS min_price_gold,
               MAX(n.price_gold)         AS max_price_gold,
               SUM(n.quantity_listed)    AS total_quantity,
               COUNT(*)                  AS obs_count
        FROM   market_observations_normalized n
        JOIN   items i ON n.item_id = i.item_id
        WHERE  n.realm_slug = ?
          AND  n.is_outlier = 0
        GROUP  BY i.archetype_id, n.realm_slug, obs_date
        ORDER  BY obs_date, i.archetype_id
        """,
        params=[realm],
    )

    written["fact_forecasts"] = _export_table(
        conn, out, "fact_forecasts", fmt,
        """
        SELECT fo.forecast_id, fo.archetype_id, fo.item_id,
               fo.realm_slug, fo.forecast_horizon, fo.target_date,
               fo.predicted_price_gold, fo.confidence_lower,
               fo.confidence_upper, fo.ci_quality, fo.model_slug,
               fo.created_at
        FROM   forecast_outputs fo
        JOIN   run_metadata rm ON fo.run_id = rm.run_id
        WHERE  fo.realm_slug = ?
        ORDER  BY fo.created_at DESC, fo.archetype_id
        """,
        params=[realm],
    )

    # Build fact_recommendations query dynamically — migration-added columns
    # (score, risk_level, score_components, category_tag) may not exist yet.
    rec_cols_available = {
        row[1] for row in conn.execute("PRAGMA table_info(recommendation_outputs)").fetchall()
    }
    rec_select = ["ro.rec_id", "ro.forecast_id", "ro.action", "ro.reasoning", "ro.priority"]
    for col in ("score", "risk_level", "score_components", "category_tag"):
        if col in rec_cols_available:
            rec_select.append(f"ro.{col}")
    rec_select.extend([
        "fo.archetype_id", "fo.realm_slug", "fo.forecast_horizon",
        "fo.predicted_price_gold", "fo.confidence_lower", "fo.confidence_upper",
    ])
    order_col = "ro.score DESC" if "score" in rec_cols_available else "ro.rec_id DESC"

    written["fact_recommendations"] = _export_table(
        conn, out, "fact_recommendations", fmt,
        f"""
        SELECT {', '.join(rec_select)}
        FROM   recommendation_outputs ro
        JOIN   forecast_outputs fo ON ro.forecast_id = fo.forecast_id
        WHERE  fo.realm_slug = ?
        ORDER  BY {order_col}
        """,
        params=[realm],
    )

    if include_backtest:
        written["fact_backtest"] = _export_table(
            conn, out, "fact_backtest", fmt,
            """
            SELECT bfr.result_id, bfr.backtest_run_id, bfr.fold_index,
                   bfr.archetype_id, bfr.realm_slug, bfr.model_name,
                   bfr.horizon_days, bfr.train_end, bfr.test_date,
                   bfr.actual_price, bfr.predicted_price,
                   bfr.abs_error, bfr.pct_error,
                   bfr.direction_correct, bfr.is_event_window
            FROM   backtest_fold_results bfr
            WHERE  bfr.realm_slug = ?
            ORDER  BY bfr.test_date, bfr.archetype_id
            """,
            params=[realm],
        )

    conn.close()
    return written


def generate_data_dictionary(
    output_path: str | Path,
) -> Path:
    """Write a markdown data dictionary documenting all BI export columns.

    Args:
        output_path: Path for the output markdown file.

    Returns:
        Path to the written file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    content = """# BI Export Data Dictionary

## Dimension Tables

### dim_archetypes
| Column | Type | Description |
|--------|------|-------------|
| archetype_id | int | Primary key |
| slug | text | URL-safe identifier (e.g. "consumable.flask.stat") |
| display_name | text | Human-readable name |
| category_tag | text | Top-level category (consumable, mat, gear, ...) |
| sub_tag | text | Subcategory within category |
| is_transferable | int | 1 if archetype transfers across expansions |
| transfer_confidence | real | Confidence score for transfer mapping (0-1) |

### dim_events
| Column | Type | Description |
|--------|------|-------------|
| event_id | int | Primary key |
| slug | text | Unique identifier |
| display_name | text | Human-readable name |
| event_type | text | Category (PATCH_RELEASE, RAID_TIER_RELEASE, etc.) |
| scope | text | GLOBAL / REGIONAL / REALM_SPECIFIC / PLAYER_SPECIFIC |
| severity | text | MINIMAL / MINOR / MODERATE / MAJOR / CRITICAL |
| expansion_slug | text | Which expansion this event belongs to |
| start_date | date | Event start date |
| end_date | date | Event end date (nullable) |

### dim_items
| Column | Type | Description |
|--------|------|-------------|
| item_id | int | Blizzard item ID (primary key) |
| name | text | In-game item name |
| category_id | int | FK to item_categories |
| archetype_id | int | FK to dim_archetypes |
| expansion_slug | text | Which expansion the item belongs to |
| quality | text | Item quality (common, uncommon, rare, epic, legendary) |
| is_crafted | int | 1 if item is crafted |
| is_boe | int | 1 if item is Bind on Equip |
| category_tag | text | Denormalized archetype category for filtering |
| archetype_name | text | Denormalized archetype display name |

### dim_dates
| Column | Type | Description |
|--------|------|-------------|
| date | date | Calendar date |
| day_of_week | int | 0=Sunday, 6=Saturday |
| day_of_year | int | 1-366 |
| week_of_year | int | ISO week number |
| month | int | 1-12 |
| is_weekend | int | 1 if Saturday or Sunday |

## Fact Tables

### fact_prices
| Column | Type | Description |
|--------|------|-------------|
| archetype_id | int | FK to dim_archetypes |
| realm_slug | text | Realm (e.g. "us") |
| obs_date | date | Observation date |
| avg_price_gold | real | Mean price in gold |
| min_price_gold | real | Minimum price |
| max_price_gold | real | Maximum price |
| total_quantity | int | Sum of listed quantity |
| obs_count | int | Number of observations |

### fact_forecasts
| Column | Type | Description |
|--------|------|-------------|
| forecast_id | int | Primary key |
| archetype_id | int | FK to dim_archetypes (null for item-level) |
| item_id | int | FK to dim_items (null for archetype-level) |
| realm_slug | text | Realm |
| forecast_horizon | text | "1d", "7d", or "28d" |
| target_date | date | Date the forecast is for |
| predicted_price_gold | real | Point estimate in gold |
| confidence_lower | real | CI lower bound |
| confidence_upper | real | CI upper bound |
| ci_quality | text | "good", "wide", or "unreliable" |
| model_slug | text | Model identifier |
| created_at | datetime | When the forecast was generated |

### fact_recommendations
| Column | Type | Description |
|--------|------|-------------|
| rec_id | int | Primary key |
| forecast_id | int | FK to fact_forecasts |
| action | text | "buy", "sell", "hold", or "avoid" |
| reasoning | text | Human-readable explanation |
| priority | int | Ranking within category |
| score | real | Composite recommendation score |
| risk_level | text | "low", "medium", "high", or "critical" |
| score_components | json | Breakdown: opp, liq, vol, event_boost, unc |
| category_tag | text | Archetype category |
| archetype_id | int | FK to dim_archetypes |
| realm_slug | text | Realm |
| forecast_horizon | text | Horizon |
| predicted_price_gold | real | Denormalized from forecast |
| confidence_lower | real | Denormalized CI lower |
| confidence_upper | real | Denormalized CI upper |

### fact_backtest (optional, include with --include-backtest)
| Column | Type | Description |
|--------|------|-------------|
| result_id | int | Primary key |
| run_id | int | Backtest run |
| fold_index | int | Walk-forward fold |
| archetype_id | int | FK to dim_archetypes |
| realm_slug | text | Realm |
| model_name | text | Model used |
| horizon_days | int | Forecast horizon in days |
| train_end | date | Training cutoff date |
| test_date | date | Date being predicted |
| actual_price | real | Actual observed price |
| predicted_price | real | Model's prediction |
| abs_error | real | |actual - predicted| |
| pct_error | real | Percentage error |
| direction_correct | int | 1 if up/down direction correct |
| is_event_window | int | 1 if an event was active |
"""
    path.write_text(content, encoding="utf-8")
    return path


# ── Internal helpers ──────────────────────────────────────────────────────────


def _export_table(
    conn: sqlite3.Connection,
    output_dir: Path,
    table_name: str,
    fmt: str,
    sql: str,
    params: list | None = None,
) -> Path:
    """Execute SQL and write results to CSV or Parquet."""
    cur = conn.execute(sql, params or [])
    columns = [desc[0] for desc in cur.description]
    rows = [dict(zip(columns, row, strict=False)) for row in cur.fetchall()]

    if fmt == "parquet":
        import pandas as pd

        path = output_dir / f"{table_name}.parquet"
        df = pd.DataFrame(rows, columns=columns)
        df.to_parquet(path, index=False)
    else:
        path = output_dir / f"{table_name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    logger.info("Exported %s: %d rows -> %s", table_name, len(rows), path)
    return path
