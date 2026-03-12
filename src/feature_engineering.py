"""
Feature Engineering Module
===========================
Creates advanced football features from cleaned match data:

- Recent form (last 5 matches)
- Rolling average goals scored / conceded
- Goal difference
- Home advantage indicator
- Shots-to-goals conversion rate
- Offensive / defensive strength
- Head-to-head statistics
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/feature_engineering.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = Path("data/processed")
FORM_WINDOW = 5  # matches to look back for form metrics


# ── Utility helpers ────────────────────────────────────────────────────────────

def _points_from_result(result: str, is_home: bool) -> int:
    """Convert a match result string to points for one side."""
    if result == "H":
        return 3 if is_home else 0
    if result == "A":
        return 0 if is_home else 3
    return 1  # Draw


def _build_team_history(df: pd.DataFrame) -> dict[str, list[dict]]:
    """
    Build a per-team chronological list of match records for rolling lookups.
    """
    history: dict[str, list[dict]] = {t: [] for t in pd.concat([df["home_team"], df["away_team"]]).unique()}

    for _, row in df.iterrows():
        ht, at = row["home_team"], row["away_team"]
        history[ht].append({
            "date": row["date"],
            "goals_scored": row["home_goals"],
            "goals_conceded": row["away_goals"],
            "shots": row.get("home_shots", np.nan),
            "shots_on": row.get("home_shots_on_target", np.nan),
            "result": row["result"],
            "is_home": True,
            "points": _points_from_result(row["result"], is_home=True),
        })
        history[at].append({
            "date": row["date"],
            "goals_scored": row["away_goals"],
            "goals_conceded": row["home_goals"],
            "shots": row.get("away_shots", np.nan),
            "shots_on": row.get("away_shots_on_target", np.nan),
            "result": row["result"],
            "is_home": False,
            "points": _points_from_result(row["result"], is_home=False),
        })

    # Keep sorted by date
    for team in history:
        history[team].sort(key=lambda x: x["date"])

    return history


def _last_n_matches(history: list[dict], before_date: pd.Timestamp, n: int) -> list[dict]:
    """Return the n most recent matches strictly before `before_date`."""
    past = [m for m in history if m["date"] < before_date]
    return past[-n:]


def _form_points(matches: list[dict]) -> float:
    """Total points from the last n matches (out of max 3*n)."""
    if not matches:
        return np.nan
    return sum(m["points"] for m in matches)


def _rolling_mean(matches: list[dict], key: str) -> float:
    vals = [m[key] for m in matches if not pd.isna(m.get(key, np.nan))]
    return float(np.mean(vals)) if vals else np.nan


# ── Feature builders ───────────────────────────────────────────────────────────

def compute_team_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each match, compute pre-match features for both home and away teams.
    """
    history = _build_team_history(df)
    records = []

    for _, row in df.iterrows():
        ht = row["home_team"]
        at = row["away_team"]
        date = row["date"]

        home_recent = _last_n_matches(history[ht], before_date=date, n=FORM_WINDOW)
        away_recent = _last_n_matches(history[at], before_date=date, n=FORM_WINDOW)

        # ── Form (points over last 5) ──────────────────────────────────────────
        home_form = _form_points(home_recent)
        away_form = _form_points(away_recent)

        # ── Rolling averages ───────────────────────────────────────────────────
        home_avg_scored     = _rolling_mean(home_recent, "goals_scored")
        home_avg_conceded   = _rolling_mean(home_recent, "goals_conceded")
        away_avg_scored     = _rolling_mean(away_recent, "goals_scored")
        away_avg_conceded   = _rolling_mean(away_recent, "goals_conceded")

        home_avg_shots      = _rolling_mean(home_recent, "shots")
        away_avg_shots      = _rolling_mean(away_recent, "shots")
        home_avg_shots_on   = _rolling_mean(home_recent, "shots_on")
        away_avg_shots_on   = _rolling_mean(away_recent, "shots_on")

        # ── Conversion rate (shots on target / shots) ─────────────────────────
        home_conversion = (
            home_avg_shots_on / home_avg_shots
            if (home_avg_shots and home_avg_shots > 0) else np.nan
        )
        away_conversion = (
            away_avg_shots_on / away_avg_shots
            if (away_avg_shots and away_avg_shots > 0) else np.nan
        )

        # ── Goal difference (rolling) ──────────────────────────────────────────
        home_gd = (
            (home_avg_scored - home_avg_conceded)
            if not any(pd.isna(x) for x in [home_avg_scored, home_avg_conceded])
            else np.nan
        )
        away_gd = (
            (away_avg_scored - away_avg_conceded)
            if not any(pd.isna(x) for x in [away_avg_scored, away_avg_conceded])
            else np.nan
        )

        # ── Head-to-head (last 5 H2H meetings) ───────────────────────────────
        h2h = df[
            (((df["home_team"] == ht) & (df["away_team"] == at)) |
             ((df["home_team"] == at) & (df["away_team"] == ht))) &
            (df["date"] < date)
        ].tail(5)

        if len(h2h):
            # From home-team perspective
            home_h2h_wins = ((h2h["home_team"] == ht) & (h2h["result"] == "H")).sum() + \
                            ((h2h["away_team"] == ht) & (h2h["result"] == "A")).sum()
            away_h2h_wins = len(h2h) - home_h2h_wins - \
                            ((h2h["result"] == "D").sum())
            h2h_win_rate  = home_h2h_wins / len(h2h)
        else:
            home_h2h_wins = away_h2h_wins = np.nan
            h2h_win_rate  = np.nan

        # ── Home advantage ────────────────────────────────────────────────────
        home_advantage = 1  # Always 1 for home side; separate feature

        record = {
            # Identifiers
            "match_id":          row.get("match_id"),
            "date":              date,
            "matchday":          row.get("matchday"),
            "home_team":         ht,
            "away_team":         at,

            # Target
            "result":            row["result"],

            # Home features
            "home_form":              home_form,
            "home_avg_scored":        home_avg_scored,
            "home_avg_conceded":      home_avg_conceded,
            "home_avg_shots":         home_avg_shots,
            "home_avg_shots_on":      home_avg_shots_on,
            "home_conversion_rate":   home_conversion,
            "home_goal_diff":         home_gd,

            # Away features
            "away_form":              away_form,
            "away_avg_scored":        away_avg_scored,
            "away_avg_conceded":      away_avg_conceded,
            "away_avg_shots":         away_avg_shots,
            "away_avg_shots_on":      away_avg_shots_on,
            "away_conversion_rate":   away_conversion,
            "away_goal_diff":         away_gd,

            # Comparative features
            "form_diff":              (home_form - away_form)
                                       if not any(pd.isna(x) for x in [home_form, away_form])
                                       else np.nan,
            "scored_diff":            (home_avg_scored - away_avg_scored)
                                       if not any(pd.isna(x) for x in [home_avg_scored, away_avg_scored])
                                       else np.nan,
            "conceded_diff":          (home_avg_conceded - away_avg_conceded)
                                       if not any(pd.isna(x) for x in [home_avg_conceded, away_avg_conceded])
                                       else np.nan,

            # H2H
            "h2h_home_wins":          home_h2h_wins,
            "h2h_away_wins":          away_h2h_wins,
            "h2h_win_rate":           h2h_win_rate,

            # Raw match stats
            "home_goals":             row["home_goals"],
            "away_goals":             row["away_goals"],
            "home_shots":             row.get("home_shots", np.nan),
            "away_shots":             row.get("away_shots", np.nan),
            "home_shots_on_target":   row.get("home_shots_on_target", np.nan),
            "away_shots_on_target":   row.get("away_shots_on_target", np.nan),
            "home_possession":        row.get("home_possession", np.nan),
            "away_possession":        row.get("away_possession", np.nan),

            # Calendar
            "home_advantage": home_advantage,
            "month":          row.get("month"),
            "day_of_week":    row.get("day_of_week"),
        }

        records.append(record)

    feature_df = pd.DataFrame(records)
    logger.info(
        "Feature matrix built: %d rows, %d columns.", *feature_df.shape
    )
    return feature_df


def impute_early_season(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """
    Impute NaN values (early-season matches lack rolling history).
    Uses column-wise median or mean imputation.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if strategy == "median":
        fill_values = df[numeric_cols].median()
    else:
        fill_values = df[numeric_cols].mean()
    # For columns where median/mean is also NaN (all-NaN columns), fill with 0
    fill_values = fill_values.fillna(0)
    df[numeric_cols] = df[numeric_cols].fillna(fill_values)
    logger.info("NaN values imputed using '%s' strategy.", strategy)
    return df


def engineer_features(clean_path: Path | None = None) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Returns
    -------
    pd.DataFrame
        Feature-rich DataFrame saved to data/processed/features.csv
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    if clean_path is None:
        clean_path = PROCESSED_DATA_DIR / "matches_clean.csv"

    df = pd.read_csv(clean_path, parse_dates=["date"])
    logger.info("Loaded clean data: %d rows", len(df))

    feature_df = compute_team_features(df)
    feature_df = impute_early_season(feature_df)

    output_path = PROCESSED_DATA_DIR / "features.csv"
    feature_df.to_csv(output_path, index=False)
    logger.info("Feature dataset saved to %s", output_path)
    return feature_df


if __name__ == "__main__":
    engineer_features()
