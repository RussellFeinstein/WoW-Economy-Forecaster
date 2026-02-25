"""
Walk-forward backtesting framework for WoW economy forecasting.

Modules
-------
splits      Walk-forward (rolling-origin) fold generation.
models      Baseline forecasting model implementations.
metrics     MAE, RMSE, MAPE, directional accuracy, and supporting types.
evaluator   Orchestrates model fitting and evaluation over all folds.
slices      Slice aggregate metrics by category, archetype, event window.
reporter    Persist results to SQLite, write CSV summaries and JSON manifest.
"""
