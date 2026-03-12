"""
Model Training Module
======================
Trains Random Forest and XGBoost classifiers on the engineered feature set.

Workflow:
  1. Time-based train/test split (last N matches held out)
  2. TimeSeriesSplit cross-validation with RandomizedSearchCV
  3. Hyperparameter tuning
  4. Model serialisation to models/
"""

import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/train_model.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
RANDOM_STATE = 42
TEST_SIZE = 0.2  # fraction of data held out as test set

# Features used for training (pre-match information only)
FEATURE_COLS = [
    "home_form", "home_avg_scored", "home_avg_conceded",
    "home_avg_shots", "home_avg_shots_on", "home_conversion_rate",
    "home_goal_diff",
    "away_form", "away_avg_scored", "away_avg_conceded",
    "away_avg_shots", "away_avg_shots_on", "away_conversion_rate",
    "away_goal_diff",
    "form_diff", "scored_diff", "conceded_diff",
    "h2h_home_wins", "h2h_away_wins", "h2h_win_rate",
    "home_advantage", "month", "day_of_week",
]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_features(path: Path | None = None) -> pd.DataFrame:
    if path is None:
        path = PROCESSED_DATA_DIR / "features.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    logger.info("Loaded features: %d rows from %s", len(df), path)
    return df


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    """Extract feature matrix X and encoded target y."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(available)
    if missing:
        logger.warning("Missing feature columns (will be skipped): %s", missing)

    X = df[available].copy()
    le = LabelEncoder()
    y = le.fit_transform(df["result"])  # A=0, D=1, H=2 (alphabetical)
    logger.info("Classes: %s", le.classes_.tolist())
    return X, pd.Series(y), le


def time_split(
    X: pd.DataFrame, y: pd.Series, test_size: float = TEST_SIZE
) -> tuple:
    """Time-ordered train / test split (no shuffling)."""
    n = len(X)
    split_idx = int(n * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    logger.info(
        "Train size: %d | Test size: %d", len(X_train), len(X_test)
    )
    return X_train, X_test, y_train, y_test


# ── Model definitions ──────────────────────────────────────────────────────────

def build_rf_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
    ])


def build_xgb_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
            n_jobs=-1,
        )),
    ])


# ── Hyperparameter grids ───────────────────────────────────────────────────────

RF_PARAM_DIST = {
    "clf__n_estimators":      [100, 200, 300, 500],
    "clf__max_depth":         [None, 5, 10, 15, 20],
    "clf__min_samples_split": [2, 5, 10],
    "clf__min_samples_leaf":  [1, 2, 4],
    "clf__max_features":      ["sqrt", "log2", 0.5],
}

XGB_PARAM_DIST = {
    "clf__n_estimators":  [100, 200, 300],
    "clf__max_depth":     [3, 5, 7, 9],
    "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "clf__subsample":     [0.6, 0.8, 1.0],
    "clf__colsample_bytree": [0.6, 0.8, 1.0],
    "clf__gamma":         [0, 0.1, 0.3],
    "clf__reg_alpha":     [0, 0.1, 0.5],
    "clf__reg_lambda":    [1, 1.5, 2],
}


# ── Training helpers ───────────────────────────────────────────────────────────

def tune_model(
    pipeline: Pipeline,
    param_dist: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 30,
    cv_splits: int = 5,
    name: str = "model",
) -> RandomizedSearchCV:
    """Run RandomizedSearchCV with TimeSeriesSplit cross-validation."""
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=tscv,
        scoring="accuracy",
        refit=True,
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    logger.info("Tuning %s (n_iter=%d, cv=%d)…", name, n_iter, cv_splits)
    search.fit(X_train, y_train)
    logger.info(
        "%s best CV accuracy: %.4f | params: %s",
        name, search.best_score_, search.best_params_,
    )
    return search


def save_model(model, name: str, label_encoder: LabelEncoder):
    """Serialise model and label encoder to models/."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / f"{name}.joblib")
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.joblib")
    logger.info("Saved %s to models/%s.joblib", name, name)


# ── Main ───────────────────────────────────────────────────────────────────────

def train(feature_path: Path | None = None) -> dict:
    """
    Full training pipeline.

    Returns
    -------
    dict
        Fitted models and metadata keyed by model name.
    """
    Path("logs").mkdir(exist_ok=True)

    df = load_features(feature_path)
    X, y, le = prepare_xy(df)
    X_train, X_test, y_train, y_test = time_split(X, y)

    results = {}

    # ── Random Forest ──────────────────────────────────────────────────────────
    rf_search = tune_model(
        build_rf_pipeline(), RF_PARAM_DIST,
        X_train, y_train,
        n_iter=20, cv_splits=5, name="RandomForest",
    )
    save_model(rf_search.best_estimator_, "random_forest", le)
    results["random_forest"] = {
        "model":       rf_search.best_estimator_,
        "cv_score":    rf_search.best_score_,
        "best_params": rf_search.best_params_,
        "X_test":      X_test,
        "y_test":      y_test,
    }

    # ── XGBoost ───────────────────────────────────────────────────────────────
    xgb_search = tune_model(
        build_xgb_pipeline(), XGB_PARAM_DIST,
        X_train, y_train,
        n_iter=20, cv_splits=5, name="XGBoost",
    )
    save_model(xgb_search.best_estimator_, "xgboost", le)
    results["xgboost"] = {
        "model":       xgb_search.best_estimator_,
        "cv_score":    xgb_search.best_score_,
        "best_params": xgb_search.best_params_,
        "X_test":      X_test,
        "y_test":      y_test,
    }

    # Persist label encoder & feature list
    joblib.dump(le, MODELS_DIR / "label_encoder.joblib")
    feature_list = [c for c in FEATURE_COLS if c in X.columns]
    joblib.dump(feature_list, MODELS_DIR / "feature_cols.joblib")
    logger.info("Training complete. Models saved to %s/", MODELS_DIR)

    return results


if __name__ == "__main__":
    train()
