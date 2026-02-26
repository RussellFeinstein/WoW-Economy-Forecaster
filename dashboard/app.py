"""
WoW Economy Forecaster — Streamlit Dashboard
=============================================

Optional local analysis UI.  Reads from ``data/outputs/`` only; it does
NOT trigger any pipeline stages or fetch new data.

Why optional?
-------------
- Streamlit adds ~100 MB of dependencies not needed for headless runs.
- The core pipeline and CLI work without it.
- All data shown here is also available via the ``wow-forecaster report-*``
  CLI commands.

App structure (5 tabs)
----------------------
  1. Top Picks     — Ranked recommendations per category with score breakdown.
  2. Forecasts     — Full forecast table with CI width; archetype detail view
                     shows a historical price line + forecast point with CI band.
  3. Volatility    — High-uncertainty items ranked by CI width.
  4. Model Health  — Drift level, MAE ratio per horizon, retrain status.
  5. Source Status — Per-source data freshness and ingestion success rate.

Provenance surfacing
--------------------
Every tab shows a colour-coded freshness badge:
  - FRESH  (green)  — report file is less than ``freshness_hours`` old.
  - STALE  (orange) — report file is older; data may not reflect current market.
  - NO DATA (red)   — no output file found; run the relevant pipeline command.

Usage
-----
    pip install -e ".[dashboard]"
    streamlit run dashboard/app.py

    # Override realm via CLI argument (passed after the double-dash):
    streamlit run dashboard/app.py -- --realm area-52
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Ensure project root is importable ────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

# ── Must be the first Streamlit call ─────────────────────────────────────────
st.set_page_config(
    page_title="WoW Economy Forecaster",
    page_icon="data/raw/snapshots",  # placeholder; Streamlit ignores missing icons
    layout="wide",
    initial_sidebar_state="expanded",
)

import pandas as pd

from dashboard.data_loader import (
    file_age_hours,
    load_drift,
    load_events,
    load_forecasts,
    load_health,
    load_historical_prices,
    load_provenance,
    load_recommendations,
)

# ── Config defaults ───────────────────────────────────────────────────────────
# These mirror config/default.toml.  Override via the sidebar or
# wow-forecaster config if your paths differ.
_DEFAULT_REALMS    = ["area-52", "illidan", "stormrage", "tichondrius"]
_REC_DIR           = str(_ROOT / "data" / "outputs" / "recommendations")
_FORECAST_DIR      = str(_ROOT / "data" / "outputs" / "forecasts")
_MON_DIR           = str(_ROOT / "data" / "outputs" / "monitoring")
_DB_PATH           = str(_ROOT / "data" / "db" / "wow_forecaster.db")
_FRESHNESS_HOURS   = 4.0


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("WoW Economy Forecaster")
    st.caption("Local analysis dashboard — reads data/outputs/ only")
    st.divider()

    realm = st.selectbox(
        "Realm",
        options=_DEFAULT_REALMS,
        index=0,
        help="Select the realm to inspect.",
    )

    freshness_hours = st.number_input(
        "Freshness threshold (hours)",
        min_value=0.5,
        max_value=72.0,
        value=_FRESHNESS_HOURS,
        step=0.5,
        help=(
            "Reports older than this are shown as STALE. "
            "Default 4 h matches the hourly refresh cadence."
        ),
    )

    if st.button("Clear cache", help="Force re-read all output files."):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption("Run pipeline to refresh data:")
    st.code("wow-forecaster run-hourly-refresh\nwow-forecaster run-daily-forecast")


# ── Freshness badge helper ────────────────────────────────────────────────────

def _freshness_badge(age: float | None, label: str = "") -> None:
    """Render a colour-coded freshness indicator using st.metric."""
    if age is None:
        st.error(f"{label}  NO DATA — run the pipeline first")
    elif age <= freshness_hours:
        st.success(f"{label}  FRESH — {age:.1f}h ago")
    else:
        st.warning(f"{label}  STALE — {age:.1f}h ago (may not reflect current market)")


def _no_data_msg(command: str) -> None:
    st.info(
        f"No data available for realm **{realm}**. "
        f"Run `wow-forecaster {command}` to generate output files."
    )


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_top, tab_fc, tab_vol, tab_health, tab_status = st.tabs(
    ["Top Picks", "Forecasts", "Volatility", "Model Health", "Source Status"]
)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Top Picks
# ══════════════════════════════════════════════════════════════════════════════

with tab_top:
    st.header("Top Recommendations by Category")
    st.caption(
        "Ranked by composite score: 0.35×opportunity + 0.20×liquidity "
        "− 0.20×volatility + 0.15×event_boost − 0.10×uncertainty"
    )

    age_rec = file_age_hours(realm, "recommendations_{realm}_*.json", _REC_DIR)
    _freshness_badge(age_rec, "Recommendations file")

    recs = load_recommendations(realm, _REC_DIR)

    if recs is None:
        _no_data_msg("run-daily-forecast")
    else:
        categories = recs.get("categories", {})
        if not categories:
            st.info("No recommendation items in the latest report.")
        else:
            # ── Category filter ────────────────────────────────────────────
            all_cats = sorted(categories.keys())
            selected_cats = st.multiselect(
                "Filter categories",
                options=all_cats,
                default=all_cats,
                help="Select one or more categories to display.",
            )

            horizon_opts = ["All horizons"] + sorted(
                {
                    item.get("horizon", "")
                    for cat in categories.values()
                    for item in cat
                }
            )
            selected_horizon = st.selectbox(
                "Horizon",
                options=horizon_opts,
                index=0,
            )

            for cat in selected_cats:
                items = categories.get(cat, [])
                if selected_horizon != "All horizons":
                    items = [i for i in items if i.get("horizon") == selected_horizon]
                if not items:
                    continue

                with st.expander(f"[{cat.upper()}] — {len(items)} item(s)", expanded=True):
                    rows = []
                    for item in items:
                        comps = item.get("score_components", {})
                        rows.append(
                            {
                                "Rank":       item.get("rank", ""),
                                "Archetype":  item.get("archetype_id", ""),
                                "Horizon":    item.get("horizon", ""),
                                "Current (g)":    item.get("current_price", ""),
                                "Predicted (g)":  item.get("predicted_price", ""),
                                "CI Low":    item.get("ci_lower", ""),
                                "CI High":   item.get("ci_upper", ""),
                                "ROI":       item.get("roi_pct", ""),
                                "Score":     item.get("score", ""),
                                "Action":    item.get("action", ""),
                                "sc_opp":    comps.get("opportunity", ""),
                                "sc_liq":    comps.get("liquidity", ""),
                                "sc_vol":    comps.get("volatility", ""),
                                "sc_evt":    comps.get("event_boost", ""),
                                "sc_unc":    comps.get("uncertainty", ""),
                                "Reasoning": item.get("reasoning", ""),
                            }
                        )
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Forecasts
# ══════════════════════════════════════════════════════════════════════════════

with tab_fc:
    st.header("Forecast Explorer")
    st.caption(
        "All forecast outputs for the selected realm. "
        "Select an archetype below to see a historical price chart with forecast overlay."
    )

    age_fc = file_age_hours(realm, "forecast_{realm}_*.csv", _FORECAST_DIR)
    _freshness_badge(age_fc, "Forecast file")

    fc_rows = load_forecasts(realm, _FORECAST_DIR)

    if not fc_rows:
        _no_data_msg("run-daily-forecast")
    else:
        df_fc = pd.DataFrame(fc_rows)

        # Coerce numeric columns.
        for col in ("current_price", "predicted_price", "ci_lower", "ci_upper", "score"):
            if col in df_fc.columns:
                df_fc[col] = pd.to_numeric(df_fc[col], errors="coerce")

        # Derived column.
        if "ci_upper" in df_fc.columns and "ci_lower" in df_fc.columns:
            df_fc["ci_width"] = df_fc["ci_upper"] - df_fc["ci_lower"]

        # ── Filters ───────────────────────────────────────────────────────
        col1, col2, col3 = st.columns(3)
        with col1:
            horizons_available = sorted(df_fc["horizon"].dropna().unique().tolist())
            sel_horizon = st.selectbox(
                "Horizon", ["All"] + horizons_available, key="fc_horiz"
            )
        with col2:
            actions_available = sorted(df_fc["action"].dropna().unique().tolist())
            sel_action = st.selectbox(
                "Action", ["All"] + actions_available, key="fc_action"
            )
        with col3:
            top_n_fc = st.number_input(
                "Show top N (by score)", min_value=5, max_value=500, value=50, key="fc_topn"
            )

        df_show = df_fc.copy()
        if sel_horizon != "All":
            df_show = df_show[df_show["horizon"] == sel_horizon]
        if sel_action != "All":
            df_show = df_show[df_show["action"] == sel_action]
        df_show = df_show.sort_values("score", ascending=False).head(int(top_n_fc))

        st.dataframe(df_show, use_container_width=True, hide_index=True)

        # ── Archetype detail + chart ───────────────────────────────────────
        st.subheader("Forecast vs Historical Price")
        st.caption(
            "Select an archetype to overlay its historical daily average price "
            "(from market_observations_normalized) with the forecast point and CI band."
        )

        archetypes = sorted(df_fc["archetype_id"].dropna().unique().tolist())
        sel_archetype = st.selectbox("Archetype", archetypes, key="fc_arch")

        if sel_archetype:
            hist = load_historical_prices(_DB_PATH, realm, sel_archetype, days=90)

            fc_arch = df_fc[df_fc["archetype_id"] == sel_archetype].copy()
            events  = load_events(_DB_PATH, days_ahead=30)

            if not hist and fc_arch.empty:
                st.info(
                    "No historical price data found for this archetype. "
                    "Run 'run-hourly-refresh' and 'build-datasets' to populate the DB."
                )
            else:
                # Build a combined DataFrame for the chart.
                chart_data: list[dict] = []

                for row in hist:
                    chart_data.append(
                        {"date": row["date"], "actual_price": row["avg_price_gold"], "predicted": None}
                    )

                for _, fc_row in fc_arch.iterrows():
                    chart_data.append(
                        {
                            "date":          fc_row.get("target_date", ""),
                            "actual_price":  None,
                            "predicted":     fc_row.get("predicted_price"),
                        }
                    )

                if chart_data:
                    df_chart = (
                        pd.DataFrame(chart_data)
                        .assign(date=lambda d: pd.to_datetime(d["date"], errors="coerce"))
                        .sort_values("date")
                        .set_index("date")
                    )
                    st.line_chart(df_chart[["actual_price", "predicted"]])

                    # Show CI bands as a separate metric row.
                    if not fc_arch.empty:
                        ci_lower_val = fc_arch["ci_lower"].iloc[0]
                        ci_upper_val = fc_arch["ci_upper"].iloc[0]
                        pred_val     = fc_arch["predicted_price"].iloc[0]
                        c1, c2, c3 = st.columns(3)
                        c1.metric("CI Lower", f"{ci_lower_val:.1f}g" if pd.notna(ci_lower_val) else "N/A")
                        c2.metric("Predicted", f"{pred_val:.1f}g" if pd.notna(pred_val) else "N/A")
                        c3.metric("CI Upper", f"{ci_upper_val:.1f}g" if pd.notna(ci_upper_val) else "N/A")

                # Event annotations (textual, below the chart).
                if events:
                    with st.expander("Nearby WoW Events (annotation reference)", expanded=False):
                        df_ev = pd.DataFrame(events)
                        st.dataframe(
                            df_ev[["display_name", "start_date", "end_date",
                                   "event_type", "severity"]],
                            use_container_width=True,
                            hide_index=True,
                        )


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Volatility Watchlist
# ══════════════════════════════════════════════════════════════════════════════

with tab_vol:
    st.header("Volatility Watchlist")
    st.caption(
        "Items ranked by CI width (ci_upper − ci_lower).  "
        "Wide CI = model is uncertain.  Treat predicted ROI with caution for these items."
    )

    _freshness_badge(age_fc, "Forecast file (same as Forecasts tab)")

    fc_vol = load_forecasts(realm, _FORECAST_DIR)

    if not fc_vol:
        _no_data_msg("run-daily-forecast")
    else:
        df_vol = pd.DataFrame(fc_vol)
        for col in ("ci_lower", "ci_upper", "predicted_price", "score"):
            if col in df_vol.columns:
                df_vol[col] = pd.to_numeric(df_vol[col], errors="coerce")

        df_vol["ci_width"] = df_vol["ci_upper"] - df_vol["ci_lower"]
        df_vol["ci_pct"]   = df_vol["ci_width"] / df_vol["predicted_price"].replace(0, float("nan"))

        top_n_vol = st.number_input(
            "Show top N most volatile", min_value=5, max_value=200, value=30
        )
        df_vol_show = df_vol.sort_values("ci_width", ascending=False).head(int(top_n_vol))

        display_cols = [
            c for c in
            ["archetype_id", "horizon", "predicted_price", "ci_lower", "ci_upper",
             "ci_width", "ci_pct", "score", "action"]
            if c in df_vol_show.columns
        ]
        st.dataframe(
            df_vol_show[display_cols],
            use_container_width=True,
            hide_index=True,
        )

        if "ci_width" in df_vol_show.columns and not df_vol_show["ci_width"].isna().all():
            st.subheader("CI Width Distribution")
            st.bar_chart(df_vol.set_index("archetype_id")["ci_width"].dropna().head(30))


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Model Health
# ══════════════════════════════════════════════════════════════════════════════

with tab_health:
    st.header("Drift & Model Health")
    st.caption(
        "Drift = distribution shift in market prices vs the 30-day baseline.  "
        "Error drift = live forecast MAE vs backtest baseline MAE.  "
        "The uncertainty multiplier widens confidence intervals in live forecasts."
    )

    age_drift  = file_age_hours(realm, "drift_status_{realm}_*.json",  _MON_DIR)
    age_health = file_age_hours(realm, "model_health_{realm}_*.json",  _MON_DIR)

    col_d, col_h = st.columns(2)
    with col_d:
        _freshness_badge(age_drift,  "Drift report")
    with col_h:
        _freshness_badge(age_health, "Health report")

    drift  = load_drift(realm,  _MON_DIR)
    health = load_health(realm, _MON_DIR)

    # ── Drift section ──────────────────────────────────────────────────────
    st.subheader("Drift Check")
    if drift is None:
        _no_data_msg("check-drift")
    else:
        overall = drift.get("overall_drift_level", "unknown").upper()
        mult    = drift.get("uncertainty_multiplier", 1.0)
        retrain = drift.get("retrain_recommended", False)

        c1, c2, c3 = st.columns(3)
        c1.metric("Overall Drift", overall)
        c2.metric("Uncertainty Mult", f"x{mult:.2f}")
        c3.metric("Retrain Needed", "YES" if retrain else "no")

        if retrain:
            st.warning(
                "Retrain recommended — run `wow-forecaster train-model` "
                "to update the model artifacts."
            )

        st.divider()
        dd = drift.get("data_drift",  {})
        ed = drift.get("error_drift", {})
        es = drift.get("event_shock", {})

        d1, d2, d3 = st.columns(3)
        d1.metric(
            "Data Drift",
            dd.get("drift_level", "N/A"),
            f"{dd.get('n_series_drifted', 0)}/{dd.get('n_series_checked', 0)} series",
        )
        mae_r = ed.get("mae_ratio")
        d2.metric(
            "Error Drift",
            ed.get("drift_level", "N/A"),
            f"MAE ratio {mae_r:.2f}x" if isinstance(mae_r, (int, float)) else "no baseline",
        )
        d3.metric(
            "Event Shock",
            "ACTIVE" if es.get("shock_active") else "none",
            f"{es.get('active_count', 0)} active, {es.get('upcoming_count', 0)} upcoming",
        )

        if es.get("active_events"):
            with st.expander("Active events causing shock", expanded=False):
                for ev in es["active_events"]:
                    slug = ev if isinstance(ev, str) else ev.get("slug", str(ev))
                    st.write(f"- {slug}")

    # ── Model health section ───────────────────────────────────────────────
    st.subheader("Model Health per Horizon")
    if health is None:
        _no_data_msg("evaluate-live-forecast")
    else:
        horizons = health.get("horizons", [])
        if not horizons:
            st.info("No horizon data in health report.")
        else:
            df_health = pd.DataFrame(horizons)
            st.dataframe(df_health, use_container_width=True, hide_index=True)

            # Colour-code health status with a bar chart of MAE ratio.
            if "mae_ratio" in df_health.columns and "horizon_days" in df_health.columns:
                df_ratio = (
                    df_health[["horizon_days", "mae_ratio"]]
                    .dropna()
                    .set_index("horizon_days")
                )
                if not df_ratio.empty:
                    st.subheader("MAE Ratio (live / baseline)")
                    st.bar_chart(df_ratio)
                    st.caption(
                        "Values > 1.5 indicate degraded performance. "
                        "> 3.0 is critical — retrain immediately."
                    )


# ══════════════════════════════════════════════════════════════════════════════
# Tab 5 — Source Status
# ══════════════════════════════════════════════════════════════════════════════

with tab_status:
    st.header("Source Status / Last Refresh")
    st.caption(
        "Data freshness from the most recent provenance report. "
        "IMPORTANT: report age != data freshness. "
        "A recently written report can still contain stale data if ingestion stopped."
    )

    age_prov = file_age_hours(realm, "provenance_{realm}_*.json", _MON_DIR)
    _freshness_badge(age_prov, "Provenance report")

    prov = load_provenance(realm, _MON_DIR)

    if prov is None:
        _no_data_msg("run-hourly-refresh")
    else:
        checked_at   = prov.get("checked_at", "?")
        freshness_h  = prov.get("freshness_hours")
        is_fresh_d   = prov.get("is_fresh", False)

        c1, c2 = st.columns(2)
        c1.metric("Report checked at", str(checked_at))
        if freshness_h is not None:
            c2.metric(
                "Data freshness",
                f"{freshness_h:.1f}h",
                "FRESH" if is_fresh_d else "STALE",
                delta_color="normal" if is_fresh_d else "inverse",
            )

        if not is_fresh_d:
            st.warning(
                "Data is stale. The most recent snapshot is older than the freshness "
                "threshold. Check your credentials (.env file) and run "
                "`wow-forecaster run-hourly-refresh`."
            )

        sources = prov.get("sources", [])
        if sources:
            st.subheader("Per-Source Details")
            df_sources = pd.DataFrame(sources)
            st.dataframe(df_sources, use_container_width=True, hide_index=True)

            # Summary metrics per source.
            for s in sources:
                source = s.get("source", "?")
                is_stale_s = s.get("is_stale", True)
                snap_count = s.get("snapshot_count_24h", 0)
                succ_rate  = s.get("success_rate_24h")

                col_s1, col_s2, col_s3 = st.columns(3)
                col_s1.metric(source, "[STALE]" if is_stale_s else "[OK]")
                col_s2.metric(f"{source} — Snaps/24h", snap_count)
                col_s3.metric(
                    f"{source} — SuccRate",
                    f"{succ_rate:.0%}" if isinstance(succ_rate, (int, float)) else "N/A",
                )
