"""
Data Cleaning Module
====================
Cleans raw match data: handles missing values, removes duplicates,
standardises team names, converts dates, and validates data types.
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
        logging.FileHandler("logs/data_cleaning.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

# Canonical team name mapping (handles common aliases / abbreviations)
TEAM_NAME_MAP: dict[str, str] = {
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Man Utd": "Manchester United",
    "Spurs": "Tottenham",
    "Tottenham Hotspur": "Tottenham",
    "Nottm Forest": "Nottingham Forest",
    "Nott'm Forest": "Nottingham Forest",
    "Newcastle": "Newcastle United",
    "Leicester": "Leicester City",
    "Ipswich": "Ipswich Town",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
    "Wolverhampton": "Wolves",
}

NUMERIC_COLS = [
    "home_goals", "away_goals",
    "home_shots", "away_shots",
    "home_shots_on_target", "away_shots_on_target",
    "home_possession", "away_possession",
    "matchday",
]


def load_raw_data(path: Path | None = None) -> pd.DataFrame:
    """Load the raw CSV produced by data_collection.py."""
    if path is None:
        path = RAW_DATA_DIR / "matches_raw.csv"
    df = pd.read_csv(path)
    logger.info("Loaded raw data: %d rows, %d columns from %s", *df.shape, path)
    return df


def standardise_team_names(df: pd.DataFrame) -> pd.DataFrame:
    """Apply canonical team name mapping."""
    for col in ("home_team", "away_team"):
        if col in df.columns:
            df[col] = df[col].str.strip().replace(TEAM_NAME_MAP)
    logger.info("Team names standardised.")
    return df


def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the date column and extract calendar features."""
    if "date" not in df.columns:
        logger.warning("No 'date' column found – skipping date conversion.")
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    invalid = df["date"].isna().sum()
    if invalid:
        logger.warning("Dropped %d rows with unparseable dates.", invalid)
        df = df.dropna(subset=["date"])

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek  # Monday=0, Sunday=6
    logger.info("Dates converted and calendar features added.")
    return df


def cast_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce expected numeric columns to float, fill NaN with median."""
    for col in NUMERIC_COLS:
        if col not in df.columns:
            logger.warning("Expected numeric column '%s' not found – skipping.", col)
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            median = df[col].median()
            df[col] = df[col].fillna(median)
            logger.debug("Filled NaN in '%s' with median=%.2f", col, median)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows (same date + teams)."""
    before = len(df)
    df = df.drop_duplicates(subset=["date", "home_team", "away_team"], keep="first")
    dropped = before - len(df)
    if dropped:
        logger.info("Removed %d duplicate rows.", dropped)
    return df


def filter_finished_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only finished matches with valid score data."""
    if "status" in df.columns:
        before = len(df)
        df = df[df["status"] == "FINISHED"]
        logger.info(
            "Kept %d finished matches (dropped %d unfinished).",
            len(df), before - len(df),
        )

    # Drop rows where both goal columns are NaN
    df = df.dropna(subset=["home_goals", "away_goals"])
    return df


def add_result_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add (or validate) the 'result' column: H / D / A."""
    if "result" not in df.columns:
        conditions = [
            df["home_goals"] > df["away_goals"],
            df["home_goals"] == df["away_goals"],
            df["home_goals"] < df["away_goals"],
        ]
        df["result"] = np.select(conditions, ["H", "D", "A"], default=np.nan)
        df = df.dropna(subset=["result"])
        logger.info("'result' column created from goal data.")
    return df


def validate_possession(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure home + away possession sums to ~100 for each row."""
    if "home_possession" in df.columns and "away_possession" in df.columns:
        total = df["home_possession"] + df["away_possession"]
        anomalous = (total - 100).abs() > 5
        if anomalous.any():
            # Normalise
            df.loc[anomalous, "home_possession"] = (
                df.loc[anomalous, "home_possession"] / total[anomalous] * 100
            ).round(1)
            df.loc[anomalous, "away_possession"] = 100 - df.loc[anomalous, "home_possession"]
            logger.info("Normalised possession for %d rows.", anomalous.sum())
    return df


def clean_data(raw_path: Path | None = None) -> pd.DataFrame:
    """
    Full cleaning pipeline.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame also saved to data/processed/matches_clean.csv
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    df = load_raw_data(raw_path)

    df = standardise_team_names(df)
    df = convert_dates(df)
    df = cast_numeric_columns(df)
    df = remove_duplicates(df)
    df = filter_finished_matches(df)
    df = add_result_column(df)
    df = validate_possession(df)

    df = df.sort_values("date").reset_index(drop=True)

    output_path = PROCESSED_DATA_DIR / "matches_clean.csv"
    df.to_csv(output_path, index=False)
    logger.info(
        "Cleaned data saved to %s (%d rows, %d columns).",
        output_path, *df.shape,
    )
    return df


if __name__ == "__main__":
    clean_data()
