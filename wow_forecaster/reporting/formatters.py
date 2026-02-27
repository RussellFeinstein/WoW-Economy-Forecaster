"""
ASCII terminal formatters for CLI reporting commands.

All formatters accept parsed report dicts / record lists and return
plain multi-line strings suitable for ``typer.echo()``.

No third-party dependencies (no ``rich``, no ``colorama``).

Freshness banners
-----------------
Every report output starts with a freshness banner so readers can tell
at a glance whether they are looking at current or stale data::

  [FRESH] Generated 1.2h ago
  [STALE] Generated 26.4h ago  <- visible warning; do not act on these
  [AGE UNKNOWN] generated_at not available

CI width (volatility)
---------------------
``format_volatility_watchlist()`` ranks items by ``ci_upper - ci_lower``
in absolute gold terms.  Wide CIs mean the model is uncertain; these
items should be treated with extra caution regardless of the predicted ROI.
"""

from __future__ import annotations


# ── Freshness banner ─────────────────────────────────────────────────────────


def format_freshness_banner(
    is_fresh: bool,
    age_hours: float | None,
    source_file: str = "",
) -> str:
    """Return a one-line freshness indicator.

    Args:
        is_fresh:    True if age <= the configured threshold.
        age_hours:   Hours since the report was generated (None = unknown).
        source_file: Optional file pattern hint (e.g. ``"drift_status_*.json"``).

    Returns:
        One or two lines of text (source_file on the second line when given).
    """
    if age_hours is None:
        tag     = "[AGE UNKNOWN]"
        age_str = "generated_at not available"
    elif is_fresh:
        tag     = "[FRESH]"
        age_str = f"Generated {age_hours:.1f}h ago"
    else:
        tag     = "[STALE]"
        age_str = f"Generated {age_hours:.1f}h ago -- data may not reflect current market"

    parts = [f"  {tag} {age_str}"]
    if source_file:
        parts.append(f"  Source: {source_file}")
    return "\n".join(parts)


# ── Top items ─────────────────────────────────────────────────────────────────


def format_top_items_table(
    categories:     dict[str, list[dict]],
    realm:          str,
    generated_at:   str,
    is_fresh:       bool,
    age_hours:      float | None,
    item_discounts: dict[int, list[dict]] | None = None,
) -> str:
    """Format top-N recommendations per category as an ASCII table.

    One block per category, sorted alphabetically.  Within each block items
    appear in rank order (already ordered by the ranker).

    When ``item_discounts`` is provided, each recommendation row is followed
    by a sub-table showing individual items within that archetype ranked by
    their price deviation from the archetype mean::

        Rank  Archetype     Horizon  Current  Predicted   ROI  Score  Action
        -------------------------------------------------------------------
           1  herb              28d   601.3g   1377.0g  +129%   52.2     buy
               Item                      Price   vs. Mean
               Luredal Kelp             210.5g    +65.0%
               Mycobloom                195.0g    +67.5%

    Args:
        categories:     ``recommendations_{realm}_{date}.json["categories"]`` dict.
        realm:          Realm slug (header).
        generated_at:   ISO string from the JSON (provenance display).
        is_fresh:       Freshness flag.
        age_hours:      Hours since generation.
        item_discounts: Optional dict keyed by archetype_id -> list of item dicts,
                        each with keys ``name``, ``item_price_gold``, ``discount_pct``.

    Returns:
        Multi-line string.
    """
    lines: list[str] = []
    lines.append("")
    lines.append("=== Top Recommendations by Category ===")
    lines.append(f"  Realm:        {realm}")
    lines.append(f"  Generated at: {generated_at}")
    lines.append(format_freshness_banner(is_fresh, age_hours))

    if not categories:
        lines.append("")
        lines.append("  (no recommendations available — run 'run-daily-forecast' first)")
        return "\n".join(lines)

    for cat in sorted(categories):
        recs = categories[cat]
        lines.append("")
        lines.append(f"  [{cat.upper()}]")
        header = (
            f"    {'Rank':>4}  {'Archetype':<30}  {'Horizon':>7}  "
            f"{'Current':>9}  {'Predicted':>9}  {'ROI':>8}  "
            f"{'Score':>6}  {'Action':>6}"
        )
        lines.append(header)
        lines.append("    " + "-" * (len(header) - 4))
        for item in recs:
            roi     = item.get("roi_pct", 0.0)
            curr    = item.get("current_price", 0.0)
            pred    = item.get("predicted_price", 0.0)
            score   = item.get("score", 0.0)
            roi_str   = f"{roi:+.1%}"    if isinstance(roi,   (int, float)) else str(roi)
            curr_str  = f"{curr:.1f}g"   if isinstance(curr,  (int, float)) else str(curr)
            pred_str  = f"{pred:.1f}g"   if isinstance(pred,  (int, float)) else str(pred)
            score_str = f"{score:.1f}"   if isinstance(score, (int, float)) else str(score)
            sub_tag   = item.get("archetype_sub_tag") or str(item.get("archetype_id", ""))
            archetype = sub_tag[:30]
            lines.append(
                f"    {item.get('rank', ''):>4}  {archetype:<30}  "
                f"{item.get('horizon', ''):>7}  {curr_str:>9}  {pred_str:>9}  "
                f"{roi_str:>8}  {score_str:>6}  {item.get('action', ''):>6}"
            )

            # Per-item discount sub-rows
            arch_id = item.get("archetype_id")
            disc_rows = (item_discounts or {}).get(arch_id, [])
            if disc_rows:
                lines.append(
                    f"          {'Item':<32}  {'Price':>9}  {'vs. Mean':>9}"
                )
                for dr in disc_rows:
                    name      = str(dr.get("name", ""))[:32]
                    price     = dr.get("item_price_gold", 0.0)
                    discount  = dr.get("discount_pct", 0.0)
                    price_str = f"{price:.1f}g"   if isinstance(price,   (int, float)) else "?"
                    disc_str  = f"{discount:+.1%}" if isinstance(discount, (int, float)) else "?"
                    lines.append(
                        f"          {name:<32}  {price_str:>9}  {disc_str:>9}"
                    )

    return "\n".join(lines)


# ── Forecast summary ──────────────────────────────────────────────────────────


def format_forecast_summary(
    records: list[dict],
    realm: str,
    top_n: int = 15,
    horizon_filter: str | None = None,
    is_fresh: bool = True,
    age_hours: float | None = None,
) -> str:
    """Format a sorted forecast summary from forecast CSV records.

    Shows the top-N rows by score, optionally filtered to one horizon.
    Columns: archetype, horizon, current price, predicted price, CI width,
    ROI, score, action.

    The CI width column surfaces forecast uncertainty directly alongside
    the predicted price so readers don't need to calculate it manually.

    Args:
        records:        List of row dicts from ``forecast_{realm}_{date}.csv``.
        realm:          Realm slug.
        top_n:          How many rows to show (full list shown on export).
        horizon_filter: Optional horizon string to filter to (e.g. ``"1d"``).
        is_fresh:       Freshness flag.
        age_hours:      Hours since the CSV was written.

    Returns:
        Multi-line string.
    """
    lines: list[str] = []
    lines.append("")
    lines.append("=== Forecast Summary ===")
    lines.append(f"  Realm:  {realm}")
    if horizon_filter:
        lines.append(f"  Horizon filter: {horizon_filter}")
    lines.append(format_freshness_banner(is_fresh, age_hours))

    filtered = records
    if horizon_filter:
        filtered = [r for r in records if r.get("horizon") == horizon_filter]

    try:
        filtered = sorted(filtered, key=lambda r: -float(r.get("score") or 0))
    except (TypeError, ValueError):
        pass

    shown = filtered[:top_n]

    if not shown:
        lines.append("")
        lines.append("  (no forecast data available — run 'run-daily-forecast' first)")
        return "\n".join(lines)

    lines.append("")
    header = (
        f"  {'Archetype':<30}  {'Horiz':>6}  {'Current':>9}  {'Pred':>9}  "
        f"{'CI Width':>9}  {'ROI':>8}  {'Score':>6}  {'Action':>6}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for r in shown:
        # CI width
        try:
            ci_lower = float(r.get("ci_lower") or 0)
            ci_upper = float(r.get("ci_upper") or 0)
            ci_str   = f"{ci_upper - ci_lower:.1f}g"
        except (TypeError, ValueError):
            ci_str = "N/A"

        # Prices
        try:
            curr_str = f"{float(r.get('current_price') or 0):.1f}g"
        except (TypeError, ValueError):
            curr_str = "N/A"
        try:
            pred_str = f"{float(r.get('predicted_price') or 0):.1f}g"
        except (TypeError, ValueError):
            pred_str = "N/A"

        # ROI — may be stored as "+5.23%" string or raw float
        roi_raw = r.get("roi_pct", "")
        if isinstance(roi_raw, str) and "%" in roi_raw:
            roi_str = roi_raw
        elif roi_raw not in ("", None):
            try:
                roi_str = f"{float(roi_raw):+.1%}"
            except (TypeError, ValueError):
                roi_str = str(roi_raw)
        else:
            roi_str = "N/A"

        try:
            score_str = f"{float(r.get('score') or 0):.1f}"
        except (TypeError, ValueError):
            score_str = "N/A"

        sub_tag = r.get("archetype_sub_tag") or str(r.get("archetype_id", ""))
        archetype = sub_tag[:30]
        lines.append(
            f"  {archetype:<30}  {r.get('horizon', ''):>6}  {curr_str:>9}  {pred_str:>9}  "
            f"{ci_str:>9}  {roi_str:>8}  {score_str:>6}  {r.get('action', ''):>6}"
        )

    total = len(filtered)
    if total > top_n:
        lines.append(
            f"  ... showing {top_n} of {total} forecasts"
            " (use --top-n N to show more, or --export to get the full set)"
        )

    return "\n".join(lines)


# ── Volatility watchlist ──────────────────────────────────────────────────────


def format_volatility_watchlist(
    records: list[dict],
    realm: str,
    top_n: int = 20,
    is_fresh: bool = True,
    age_hours: float | None = None,
) -> str:
    """Format a volatility watchlist: items ranked by CI width (widest first).

    CI width = ``ci_upper - ci_lower`` in gold.  Wide CI means the model is
    uncertain — the actual price could deviate significantly from the
    prediction.  These items carry the most risk regardless of predicted ROI.

    A separate ``CI %`` column shows CI width relative to predicted price,
    so high-value and low-value items can be compared fairly.

    Args:
        records:  Forecast CSV rows.
        realm:    Realm slug.
        top_n:    Number of items to show.
        is_fresh: Freshness flag.
        age_hours: Hours since CSV was written.

    Returns:
        Multi-line string.
    """
    lines: list[str] = []
    lines.append("")
    lines.append("=== Volatility Watchlist ===")
    lines.append(f"  Realm: {realm}")
    lines.append("  Items ranked by CI width (highest forecast uncertainty first).")
    lines.append("  Wide CI = model is unsure; treat predicted ROI with caution.")
    lines.append(format_freshness_banner(is_fresh, age_hours))

    enriched: list[dict] = []
    for r in records:
        try:
            ci_lower = float(r.get("ci_lower") or 0)
            ci_upper = float(r.get("ci_upper") or 0)
            pred     = float(r.get("predicted_price") or 0)
            ci_width = ci_upper - ci_lower
            ci_pct   = ci_width / pred if pred > 0 else 0.0
            enriched.append({**r, "_ci_width": ci_width, "_ci_pct": ci_pct})
        except (TypeError, ValueError):
            continue

    enriched.sort(key=lambda r: -r["_ci_width"])
    shown = enriched[:top_n]

    if not shown:
        lines.append("")
        lines.append("  (no forecast data available — run 'run-daily-forecast' first)")
        return "\n".join(lines)

    lines.append("")
    header = (
        f"  {'Archetype':<30}  {'Horiz':>6}  {'Predicted':>9}  "
        f"{'CI Width':>9}  {'CI %':>7}  {'Score':>6}  {'Action':>6}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for r in shown:
        pred_str   = f"{float(r.get('predicted_price') or 0):.1f}g"
        ci_str     = f"{r['_ci_width']:.1f}g"
        ci_pct_str = f"{r['_ci_pct']:.0%}"
        try:
            score_str = f"{float(r.get('score') or 0):.1f}"
        except (TypeError, ValueError):
            score_str = "N/A"
        sub_tag = r.get("archetype_sub_tag") or str(r.get("archetype_id", ""))
        archetype = sub_tag[:30]
        lines.append(
            f"  {archetype:<30}  {r.get('horizon', ''):>6}  {pred_str:>9}  "
            f"{ci_str:>9}  {ci_pct_str:>7}  {score_str:>6}  {r.get('action', ''):>6}"
        )

    total = len(enriched)
    if total > top_n:
        lines.append(
            f"  ... showing {top_n} of {total} (use --top-n N or --export for full list)"
        )

    return "\n".join(lines)


# ── Drift & model health ──────────────────────────────────────────────────────


def format_drift_health_summary(
    drift: dict | None,
    health: dict | None,
    realm: str,
    is_fresh_drift: bool = True,
    age_hours_drift: float | None = None,
    is_fresh_health: bool = True,
    age_hours_health: float | None = None,
) -> str:
    """Format a combined drift + model health summary.

    Two sections in one output:

    **Drift section** — data drift level, error drift, event shock flag,
    uncertainty multiplier.  The multiplier is what actually affects live
    forecast confidence intervals, so it is prominent.

    **Model health section** — per-horizon live MAE vs backtest baseline,
    health status.  The ``mae_ratio`` column makes degradation immediately
    visible: a value of ``2.0x`` means the live model is twice as wrong
    as the backtest baseline.

    Args:
        drift:            Parsed ``drift_status_{realm}_{date}.json``.
        health:           Parsed ``model_health_{realm}_{date}.json``.
        realm:            Realm slug.
        is_fresh_drift:   Freshness for drift report.
        age_hours_drift:  Age of drift report in hours.
        is_fresh_health:  Freshness for health report.
        age_hours_health: Age of health report in hours.

    Returns:
        Multi-line string.
    """
    lines: list[str] = []
    lines.append("")
    lines.append("=== Drift & Model Health ===")
    lines.append(f"  Realm: {realm}")

    # ── Drift section ─────────────────────────────────────────────────────────
    lines.append("")
    lines.append("  ---- Drift Check ----")
    if drift is None:
        lines.append("  (no drift report available -- run 'check-drift' first)")
    else:
        lines.append(
            format_freshness_banner(is_fresh_drift, age_hours_drift, "drift_status_*.json")
        )
        lines.append("")

        overall = drift.get("overall_drift_level", "unknown")
        mult    = drift.get("uncertainty_multiplier", 1.0)
        retrain = drift.get("retrain_recommended", False)

        lines.append(f"  Overall drift:       {overall.upper()}")
        lines.append(f"  Uncertainty mult:    x{mult:.2f}")
        retrain_msg = "YES  <-- ACTION REQUIRED: run 'train-model'" if retrain else "no"
        lines.append(f"  Retrain recommended: {retrain_msg}")

        dd = drift.get("data_drift", {})
        ed = drift.get("error_drift", {})
        es = drift.get("event_shock", {})

        lines.append("")
        dd_series = (
            f"({dd.get('n_series_drifted', '?')}/{dd.get('n_series_checked', '?')} series)"
        )
        lines.append(f"  Data drift:  {dd.get('drift_level', 'N/A'):<8}  {dd_series}")

        mae_ratio = ed.get("mae_ratio")
        mae_tag   = (
            f"  MAE ratio={mae_ratio:.2f}x" if isinstance(mae_ratio, (int, float)) else ""
        )
        lines.append(f"  Error drift: {ed.get('drift_level', 'N/A'):<8}{mae_tag}")

        shock_flag = es.get("shock_active", False)
        active_n   = es.get("active_count", 0)
        upcoming_n = es.get("upcoming_count", 0)
        shock_str  = "ACTIVE" if shock_flag else "none"
        lines.append(
            f"  Event shock: {shock_str:<8}  ({active_n} active, {upcoming_n} upcoming)"
        )
        if es.get("active_events"):
            for ev in es["active_events"][:3]:
                slug = ev if isinstance(ev, str) else ev.get("slug", str(ev))
                lines.append(f"    -> {slug}")

    # ── Model health section ───────────────────────────────────────────────────
    lines.append("")
    lines.append("  ---- Model Health ----")
    if health is None:
        lines.append(
            "  (no health report available -- run 'evaluate-live-forecast' first)"
        )
    else:
        lines.append(
            format_freshness_banner(is_fresh_health, age_hours_health, "model_health_*.json")
        )
        lines.append("")
        horizons = health.get("horizons", [])
        if not horizons:
            lines.append("  (no horizon data in health report)")
        else:
            header = (
                f"  {'H':>5}  {'Status':>10}  {'N':>5}  "
                f"{'LiveMAE':>9}  {'BaseMAE':>9}  {'Ratio':>7}  {'DirAcc':>7}"
            )
            lines.append(header)
            lines.append("  " + "-" * (len(header) - 2))
            for h in horizons:
                h_days   = h.get("horizon_days", "?")
                status   = h.get("health_status", "unknown")
                n_eval   = h.get("n_evaluated", 0)
                live_mae = h.get("live_mae")
                base_mae = h.get("baseline_mae")
                ratio    = h.get("mae_ratio")
                dir_acc  = h.get("live_dir_acc")

                live_s  = f"{live_mae:.2f}g" if isinstance(live_mae, (int, float)) else "N/A"
                base_s  = f"{base_mae:.2f}g" if isinstance(base_mae, (int, float)) else "N/A"
                ratio_s = f"{ratio:.2f}x"    if isinstance(ratio,    (int, float)) else "N/A"
                dir_s   = f"{dir_acc:.1%}"   if isinstance(dir_acc,  (int, float)) else "N/A"

                lines.append(
                    f"  {h_days:>4}d  {status:>10}  {n_eval:>5}  "
                    f"{live_s:>9}  {base_s:>9}  {ratio_s:>7}  {dir_s:>7}"
                )

    return "\n".join(lines)


# ── Source status ─────────────────────────────────────────────────────────────


def format_status_summary(
    provenance: dict | None,
    realm: str,
    is_fresh_prov: bool = True,
    age_hours_prov: float | None = None,
) -> str:
    """Format a source freshness and last-refresh summary.

    Operations-level view — shows the three data sources (undermine,
    blizzard_api, blizzard_news) with per-source snapshot counts,
    record counts, and success rates from the last 24 hours.

    Distinguishes between:
    - **Report age** — how old the provenance file itself is.
    - **Data freshness** — how old the most recent snapshot inside the
      report is (``freshness_hours``).  A recently written provenance file
      can still report stale data if the ingestion pipeline stopped.

    This distinction is surfaced explicitly to prevent a user from
    thinking "the file was written 2 minutes ago so the data is fresh".

    Args:
        provenance:      Parsed ``provenance_{realm}_{date}.json``.
        realm:           Realm slug.
        is_fresh_prov:   Whether the provenance file itself is recent.
        age_hours_prov:  Age of the provenance file in hours.

    Returns:
        Multi-line string.
    """
    lines: list[str] = []
    lines.append("")
    lines.append("=== Source Status / Last Refresh ===")
    lines.append(f"  Realm: {realm}")

    if provenance is None:
        lines.append(
            "  (no provenance report available -- run 'run-hourly-refresh' first)"
        )
        return "\n".join(lines)

    lines.append(format_freshness_banner(is_fresh_prov, age_hours_prov, "provenance_*.json"))
    lines.append("")

    checked_at   = provenance.get("checked_at", "?")
    freshness_h  = provenance.get("freshness_hours")
    is_fresh_d   = provenance.get("is_fresh", False)

    lines.append(f"  Report checked at:  {checked_at}")
    if freshness_h is not None:
        fresh_tag = "FRESH" if is_fresh_d else "STALE -- no recent snapshots"
        lines.append(f"  Data freshness:     {freshness_h:.1f}h  [{fresh_tag}]")
        lines.append(
            "  Note: report age != data freshness. "
            "A recent report can contain stale data if ingestion stopped."
        )

    sources = provenance.get("sources", [])
    if not sources:
        lines.append("  (no per-source entries in report)")
        return "\n".join(lines)

    lines.append("")
    header = (
        f"  {'Source':<20}  {'Last Snapshot':<22}  "
        f"{'Snaps/24h':>9}  {'Records':>9}  {'SuccRate':>9}  {'Status':>8}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for s in sources:
        source     = s.get("source", "?")
        last_snap  = s.get("last_snapshot_at") or "never"
        snap_count = s.get("snapshot_count_24h", 0)
        rec_count  = s.get("total_records_24h", 0)
        succ_rate  = s.get("success_rate_24h")
        is_stale   = s.get("is_stale", True)

        succ_s = f"{succ_rate:.0%}" if isinstance(succ_rate, (int, float)) else "N/A"
        status = "[STALE]" if is_stale else "[OK]"

        if isinstance(last_snap, str) and len(last_snap) > 22:
            last_snap = last_snap[:19] + "..."

        lines.append(
            f"  {source:<20}  {last_snap:<22}  "
            f"{snap_count:>9}  {rec_count:>9}  {succ_s:>9}  {status:>8}"
        )

    return "\n".join(lines)
