"""
ML forecasting layer — LightGBM-based price forecasting for WoW AH data.

Modules
-------
feature_selector  : Defines which Parquet columns are model inputs; encodes
                    categorical/bool fields to integers for LightGBM.
lgbm_model        : LightGBMForecaster class (fit, predict, save, load).
cold_start        : Heuristic confidence-interval widening for Midnight items
                    with insufficient history.
trainer           : train_models() orchestrator — one model per horizon,
                    time-based validation split, artifact persistence.
predictor         : run_inference() — loads model + inference Parquet,
                    produces ForecastOutput objects.
"""
