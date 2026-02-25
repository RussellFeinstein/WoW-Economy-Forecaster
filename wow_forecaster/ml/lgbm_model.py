"""
LightGBM-based price forecaster.

Model choice rationale
----------------------
LightGBM is chosen over alternatives:

  vs XGBoost:    Histogram-based leaf-wise growth is faster and uses less
                 memory. Marginal accuracy differences on tabular data.

  vs Prophet:    Prophet requires 30+ observations per series for reliable
                 seasonality. Cold-start Midnight archetypes have fewer.
                 Prophet cannot consume cross-series features (event severity,
                 archetype category) that distinguish WoW market behaviour.

  vs LSTM/RNN:   Needs 100+ timesteps per series; harder to deploy at
                 inference time; no native feature interaction support.

Training strategy
-----------------
ONE global model per forecast horizon across ALL archetype-realm series.
Cross-archetype learning benefits:
  - "Consumables spike before major patch events" generalises across items.
  - Cold-start archetypes in the training data teach the model to be
    conservative when is_cold_start_int=1.
  - Sufficient training data even when individual series are thin.

archetype_id is deliberately EXCLUDED as a feature — using it would memorise
specific TWW archetype IDs and prevent transfer to new Midnight item IDs.

Validation split
----------------
Always time-based (last N days = validation). NEVER random — random splits
on time-series data allow the model to peek into the future.

Missing values
--------------
LightGBM handles NaN natively. All Parquet Nones become float("nan")
before matrix construction; never pass Python None to lgb.Dataset.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class LightGBMForecaster:
    """Global LightGBM price forecaster for all archetypes in a realm.

    Attributes:
        horizon_days: Forecast horizon this model was trained for.
        MODEL_VERSION: Package-level version string embedded in artifact metadata.
    """

    MODEL_VERSION = "v0.5.0"

    def __init__(
        self,
        horizon_days: int,
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        n_estimators: int = 100,
        min_child_samples: int = 5,
        feature_fraction: float = 0.8,
        bagging_fraction: float = 0.8,
        bagging_freq: int = 5,
        early_stopping_rounds: int = 20,
    ) -> None:
        self.horizon_days = horizon_days
        self._hyperparams: dict[str, Any] = {
            "num_leaves":        num_leaves,
            "learning_rate":     learning_rate,
            "n_estimators":      n_estimators,
            "min_child_samples": min_child_samples,
            "feature_fraction":  feature_fraction,
            "bagging_fraction":  bagging_fraction,
            "bagging_freq":      bagging_freq,
        }
        self._early_stopping_rounds = early_stopping_rounds
        self._booster = None       # lgb.Booster; None until fit()
        self._feature_cols: list[str] = []
        self._categorical_indices: list[int] = []
        self._val_metrics: dict[str, float] = {}
        self._training_rows: int = 0
        self._trained_at: str = ""

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        """True after fit() has been called successfully."""
        return self._booster is not None

    @property
    def val_metrics(self) -> dict[str, float]:
        """Validation-set metrics from the most recent fit() call."""
        return dict(self._val_metrics)

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        train_rows: list[dict[str, Any]],
        val_rows: list[dict[str, Any]],
        feature_cols: list[str],
        categorical_cols: list[str],
        target_col: str,
    ) -> dict[str, float]:
        """Train on training rows and evaluate on validation rows.

        Args:
            train_rows: Pre-encoded training rows (from encode_row()).
            val_rows:   Pre-encoded validation rows (may be empty).
            feature_cols:   Feature column names in TRAINING_FEATURE_COLS order.
            categorical_cols: Subset of feature_cols that are categorical ints.
            target_col: Name of the target column (e.g. "target_price_7d").

        Returns:
            Dict of validation metrics: mae, rmse, n_val (and mape if non-zero
            actual prices exist).

        Raises:
            ValueError: Fewer than 10 training rows, or no valid target labels.
        """
        import lightgbm as lgb

        from wow_forecaster.ml.feature_selector import build_feature_matrix, to_float

        if len(train_rows) < 10:
            raise ValueError(
                f"LightGBMForecaster.fit() needs >= 10 training rows; "
                f"got {len(train_rows)}."
            )

        self._feature_cols = list(feature_cols)
        self._categorical_indices = [
            i for i, c in enumerate(feature_cols) if c in categorical_cols
        ]

        train_valid = [r for r in train_rows if r.get(target_col) is not None]
        val_valid   = [r for r in val_rows   if r.get(target_col) is not None]

        if not train_valid:
            raise ValueError(
                f"No rows have a non-null target '{target_col}'. "
                "Ensure the training Parquet includes forward-looking labels."
            )

        import numpy as np

        X_train = np.array(build_feature_matrix(train_valid, feature_cols), dtype=np.float64)
        y_train = np.array([to_float(r[target_col]) for r in train_valid], dtype=np.float64)
        X_val   = np.array(build_feature_matrix(val_valid, feature_cols), dtype=np.float64) if val_valid else np.empty((0, len(feature_cols)), dtype=np.float64)
        y_val   = np.array([to_float(r[target_col]) for r in val_valid], dtype=np.float64)  if val_valid else np.array([], dtype=np.float64)

        lgb_params = {
            "objective":        "regression_l1",  # MAE loss — robust to outlier prices
            "metric":           "mae",
            "num_leaves":       self._hyperparams["num_leaves"],
            "learning_rate":    self._hyperparams["learning_rate"],
            "feature_fraction": self._hyperparams["feature_fraction"],
            "bagging_fraction": self._hyperparams["bagging_fraction"],
            "bagging_freq":     self._hyperparams["bagging_freq"],
            "min_child_samples":self._hyperparams["min_child_samples"],
            "verbose":          -1,
            "n_jobs":           -1,
        }

        dtrain = lgb.Dataset(
            X_train,
            label=y_train,
            feature_name=feature_cols,
            categorical_feature=self._categorical_indices,
            free_raw_data=False,
        )

        callbacks = [lgb.log_evaluation(period=-1)]
        valid_sets  = [dtrain]
        valid_names = ["train"]

        if val_valid:
            dval = lgb.Dataset(
                X_val,
                label=y_val,
                feature_name=feature_cols,
                categorical_feature=self._categorical_indices,
                reference=dtrain,
                free_raw_data=False,
            )
            valid_sets  = [dtrain, dval]
            valid_names = ["train", "val"]
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=self._early_stopping_rounds,
                    verbose=False,
                )
            )

        self._booster = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=self._hyperparams["n_estimators"],
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        self._training_rows = len(train_valid)
        self._trained_at    = date.today().isoformat()

        if val_valid:
            self._val_metrics = self._evaluate(X_val, y_val)
        else:
            self._val_metrics = {}

        return dict(self._val_metrics)

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, rows: list[dict[str, Any]]) -> list[float | None]:
        """Predict price for a list of pre-encoded inference rows.

        Args:
            rows: Pre-encoded inference rows (from encode_row()).

        Returns:
            List of predicted gold prices (same length as rows).
            Returns ``[None] * len(rows)`` if the model is not fitted.
        """
        if not self.is_fitted or not rows:
            return [None] * len(rows)

        import numpy as np

        from wow_forecaster.ml.feature_selector import build_feature_matrix

        X = np.array(build_feature_matrix(rows, self._feature_cols), dtype=np.float64)
        preds = self._booster.predict(X)
        return [max(0.0, float(p)) for p in preds]

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _evaluate(self, X: list[list[float]], y: list[float]) -> dict[str, float]:
        """Compute MAE, RMSE, MAPE on a feature matrix + true labels."""
        preds = self._booster.predict(X)
        n = len(y)
        if n == 0:
            return {}
        mae_sum = rmse_sum = mape_sum = 0.0
        mape_n = 0
        for actual, pred in zip(y, preds):
            err = actual - pred
            mae_sum  += abs(err)
            rmse_sum += err * err
            if actual != 0.0:
                mape_sum += abs(err / actual)
                mape_n   += 1
        metrics: dict[str, float] = {
            "mae":   mae_sum / n,
            "rmse":  math.sqrt(rmse_sum / n),
            "n_val": float(n),
        }
        if mape_n > 0:
            metrics["mape"] = mape_sum / mape_n
        return metrics

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, artifact_path: Path) -> None:
        """Serialize the booster to a joblib pickle file.

        Args:
            artifact_path: Target .pkl path. Parent directories are created.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted LightGBMForecaster.")

        import joblib

        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "booster":               self._booster,
                "feature_cols":          self._feature_cols,
                "categorical_indices":   self._categorical_indices,
                "hyperparams":           self._hyperparams,
                "val_metrics":           self._val_metrics,
                "training_rows":         self._training_rows,
                "horizon_days":          self.horizon_days,
                "model_version":         self.MODEL_VERSION,
                "trained_at":            self._trained_at,
            },
            artifact_path,
        )
        logger.info("Model artifact saved: %s", artifact_path)

    @classmethod
    def load(cls, artifact_path: Path) -> "LightGBMForecaster":
        """Load a serialized LightGBMForecaster from disk.

        Args:
            artifact_path: Path to a .pkl file written by save().

        Returns:
            A fitted LightGBMForecaster instance.

        Raises:
            FileNotFoundError: If artifact_path does not exist.
        """
        import joblib

        if not artifact_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {artifact_path}")

        state = joblib.load(artifact_path)
        hyp = state.get("hyperparams", {})
        inst = cls(horizon_days=state["horizon_days"], **hyp)
        inst._booster              = state["booster"]
        inst._feature_cols         = state["feature_cols"]
        inst._categorical_indices  = state.get("categorical_indices", [])
        inst._val_metrics          = state.get("val_metrics", {})
        inst._training_rows        = state.get("training_rows", 0)
        inst._trained_at           = state.get("trained_at", "")
        logger.info(
            "Model artifact loaded: %s (horizon=%dd, trained=%s)",
            artifact_path, inst.horizon_days, inst._trained_at,
        )
        return inst

    def write_metadata(
        self,
        meta_path: Path,
        realm_slug: str,
        dataset_version: str,
    ) -> None:
        """Write a JSON metadata sidecar alongside the model artifact.

        Args:
            meta_path:       Destination .json path.
            realm_slug:      Realm this model was trained for.
            dataset_version: Filename of the training Parquet (provenance).
        """
        from wow_forecaster.ml.feature_selector import TARGET_COL_MAP

        meta = {
            "schema_version":      self.MODEL_VERSION,
            "model_type":          "lightgbm",
            "horizon_days":        self.horizon_days,
            "realm_slug":          realm_slug,
            "trained_at":          self._trained_at,
            "dataset_version":     dataset_version,
            "feature_columns":     self._feature_cols,
            "target_column":       TARGET_COL_MAP.get(
                self.horizon_days, f"target_price_{self.horizon_days}d"
            ),
            "hyperparameters":     self._hyperparams,
            "validation_metrics":  self._val_metrics,
            "training_rows":       self._training_rows,
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        logger.debug("Model metadata written: %s", meta_path)
