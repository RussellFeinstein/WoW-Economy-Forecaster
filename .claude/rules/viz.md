---
paths:
  - "wow_forecaster/viz/**"
  - "dashboard/**"
  - "notebooks/**"
  - "wow_forecaster/reporting/bi_export.py"
---

# Visualization and portfolio

Loaded when working with charts, the dashboard, BI exports, or notebooks. Root context: [CLAUDE.md](../../CLAUDE.md).

## Visualization & Portfolio (v2.2.0)
- [wow_forecaster/viz/](../../wow_forecaster/viz/) — publication-quality chart layer (matplotlib/seaborn/Plotly)
- [wow_forecaster/viz/theme.py](../../wow_forecaster/viz/theme.py) — WoW dark palette, apply_wow_theme(), get_plotly_template()
- [wow_forecaster/viz/data_queries.py](../../wow_forecaster/viz/data_queries.py) — SQL/file -> pandas DataFrame fetchers
- [wow_forecaster/viz/charts/](../../wow_forecaster/viz/charts/) — 6 chart modules: forecast, backtest, feature, recommendation, drift, transfer
- [wow_forecaster/reporting/bi_export.py](../../wow_forecaster/reporting/bi_export.py) — Star-schema dim/fact table exports for Power BI / Tableau
- CLI: generate-charts (--chart-type, --format png|svg|both), export-bi-bundle (--format csv|parquet)
- Optional dep group: `[viz]` (matplotlib, seaborn, plotly, kaleido, pandas); `[dashboard]` now depends on `[viz]`
- Dashboard upgraded to 8 tabs (added Backtest, Feature Insights, Crafting); Plotly interactive forecast chart
- 3 Jupyter analysis notebooks in notebooks/ (EDA, Model Development, Backtest Evaluation)
- GitHub Actions CI workflow (.github/workflows/ci.yml)
