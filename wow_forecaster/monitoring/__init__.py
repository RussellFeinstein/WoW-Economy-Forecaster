"""
WoW Economy Forecaster — Monitoring package.

Provides drift detection, adaptive policy, model health, and provenance
tracking for the hourly refresh orchestration pipeline.

Modules:
    drift      — Data drift, error drift, and event-shock detection.
    adaptive   — Adaptive uncertainty policy based on drift severity.
    health     — Live model performance evaluation (MAE vs baseline).
    provenance — Source freshness and attribution tracking.
    reporter   — Write monitoring outputs to disk (JSON).
"""
