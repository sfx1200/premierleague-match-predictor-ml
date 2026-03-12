"""
Prediction Utility
==================
Loads trained models and generates predictions for arbitrary match-ups.
Used by both the FastAPI layer and the Streamlit dashboard.
"""

import logging
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODELS_DIR    = Path("models")
PROCESSED_DIR = Path("data/processed")


class MatchPredictor:
    """Encapsulates model loading and inference."""

    def __init__(self, model_name: str = "xgboost"):
        self.model_name = model_name
        self.model      = joblib.load(MODELS_DIR / f"{model_name}.joblib")
        self.le         = joblib.load(MODELS_DIR / "label_encoder.joblib")
        self.feat_cols  = joblib.load(MODELS_DIR / "feature_cols.joblib")
        self._feature_df = self._load_feature_df()
        logger.info("MatchPredictor loaded model: %s", model_name)

    def _load_feature_df(self) -> pd.DataFrame:
        path = PROCESSED_DIR / "features.csv"
        if path.exists():
            return pd.read_csv(path, parse_dates=["date"])
        return pd.DataFrame()

    def _get_team_features(self, team: str, is_home: bool) -> dict:
        """Extract the most recent feature snapshot for a given team."""
        df = self._feature_df
        if df.empty:
            return {}

        prefix = "home" if is_home else "away"
        team_col = f"{prefix}_team"
        df_team = df[df[team_col] == team].sort_values("date")

        if df_team.empty:
            logger.warning("No historical data found for team: %s", team)
            return {}

        latest = df_team.iloc[-1]
        snapshot = {}
        for col in self.feat_cols:
            if col.startswith(prefix + "_"):
                snapshot[col] = latest.get(col, np.nan)
        return snapshot

    def _build_feature_vector(self, home_team: str, away_team: str) -> pd.DataFrame:
        home_feats = self._get_team_features(home_team, is_home=True)
        away_feats = self._get_team_features(away_team, is_home=False)

        row = {**home_feats, **away_feats}

        # Comparative features
        try:
            row["form_diff"]    = row.get("home_form", 0) - row.get("away_form", 0)
            row["scored_diff"]  = row.get("home_avg_scored", 0) - row.get("away_avg_scored", 0)
            row["conceded_diff"]= row.get("home_avg_conceded", 0) - row.get("away_avg_conceded", 0)
        except Exception:
            pass

        # Constants
        row["home_advantage"] = 1

        # H2H defaults
        for col in ("h2h_home_wins", "h2h_away_wins", "h2h_win_rate"):
            if col not in row:
                row[col] = np.nan

        # Temporal
        import datetime
        now = datetime.datetime.now()
        row["month"]       = now.month
        row["day_of_week"] = now.weekday()

        vec = pd.DataFrame([row])
        # Align columns
        for c in self.feat_cols:
            if c not in vec.columns:
                vec[c] = np.nan
        vec = vec[self.feat_cols]
        vec = vec.fillna(vec.median(numeric_only=True))
        return vec

    def predict(self, home_team: str, away_team: str) -> dict:
        """
        Predict match outcome.

        Returns
        -------
        dict with keys: home_win, draw, away_win, predicted_outcome
        """
        X = self._build_feature_vector(home_team, away_team)
        probs = self.model.predict_proba(X)[0]
        classes = self.le.classes_  # ['A', 'D', 'H']

        prob_map = dict(zip(classes, probs))
        predicted_class = classes[np.argmax(probs)]
        outcome_map = {"H": "Home Win", "D": "Draw", "A": "Away Win"}

        return {
            "home_team":         home_team,
            "away_team":         away_team,
            "home_win":          round(float(prob_map.get("H", 0)), 4),
            "draw":              round(float(prob_map.get("D", 0)), 4),
            "away_win":          round(float(prob_map.get("A", 0)), 4),
            "predicted_outcome": outcome_map.get(predicted_class, predicted_class),
            "model_used":        self.model_name,
        }


# Singleton cache
_predictor_cache: dict[str, MatchPredictor] = {}


def get_predictor(model_name: str = "xgboost") -> MatchPredictor:
    global _predictor_cache
    if model_name not in _predictor_cache:
        _predictor_cache[model_name] = MatchPredictor(model_name)
    return _predictor_cache[model_name]


if __name__ == "__main__":
    predictor = get_predictor("xgboost")
    result = predictor.predict("Arsenal", "Liverpool")
    print(result)
