"""
Recommendation engine: converts LightGBM forecasts into ranked
buy/sell/hold/avoid recommendations with human-readable explanations.

Modules
-------
scorer   : ScoreComponents dataclass + compute_score() + determine_action()
           + build_reasoning() — pure functions, no DB or I/O.
ranker   : ScoredForecast dataclass + build_scored_forecasts() +
           top_n_per_category() + build_recommendation_outputs().
reporter : write_forecast_csv() + write_recommendation_json() — file output.
"""
