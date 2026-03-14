"""
Data Collection Module
======================
Fetches Premier League 2025-2026 match data from football APIs.
Falls back to generating realistic synthetic data when no API key is available.
"""

import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Configure logging
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/data_collection.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
RAW_DATA_DIR = Path("data/raw")
SEASON = "2025"
LEAGUE_ID = 39  # Premier League on api-football

PL_TEAMS_2025 = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Burnley", "Chelsea", "Crystal Palace", "Everton", "Fulham",
    "Leeds United", "Liverpool", "Manchester City", "Manchester United",
    "Newcastle United", "Nottingham Forest", "Sunderland", "Tottenham",
    "West Ham", "Wolves",
]


# ── API Client ─────────────────────────────────────────────────────────────────

class FootballDataClient:
    """Client for football-data.org free API (no key required for basic use)."""

    BASE_URL = "https://api.football-data.org/v4"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("FOOTBALL_DATA_API_KEY", "")
        self.headers = {"X-Auth-Token": self.api_key} if self.api_key else {}
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def get_matches(self, season: str = "2025") -> list[dict]:
        """Fetch all PL matches for the given season."""
        url = f"{self.BASE_URL}/competitions/PL/matches"
        params = {"season": season}
        try:
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            logger.info("Fetched %d matches from API", len(data.get("matches", [])))
            return data.get("matches", [])
        except requests.RequestException as exc:
            logger.warning("API request failed: %s", exc)
            return []

    def parse_matches(self, raw_matches: list[dict]) -> pd.DataFrame:
        """Parse raw API response into a clean DataFrame."""
        records = []
        for m in raw_matches:
            score = m.get("score", {})
            full_time = score.get("fullTime", {})
            home_goals = full_time.get("home")
            away_goals = full_time.get("away")

            records.append({
                "match_id": m.get("id"),
                "date": m.get("utcDate", "")[:10],
                "home_team": m["homeTeam"]["name"],
                "away_team": m["awayTeam"]["name"],
                "home_goals": home_goals,
                "away_goals": away_goals,
                "status": m.get("status"),
                "matchday": m.get("matchday"),
                "stage": m.get("stage"),
            })

        df = pd.DataFrame(records)
        logger.info("Parsed %d match records", len(df))
        return df


# ── Synthetic Data Generator ───────────────────────────────────────────────────

class SyntheticDataGenerator:
    """
    Generates realistic synthetic Premier League data for the 2025-26 season.
    Used when no live API key is available.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        # Relative strengths (higher = stronger attack/defense)
        self.team_strength = {
            "Manchester City":  {"attack": 0.95, "defense": 0.90},
            "Liverpool":        {"attack": 0.93, "defense": 0.89},
            "Arsenal":          {"attack": 0.89, "defense": 0.88},
            "Chelsea":          {"attack": 0.83, "defense": 0.80},
            "Tottenham":        {"attack": 0.81, "defense": 0.75},
            "Manchester United": {"attack": 0.78, "defense": 0.74},
            "Newcastle United": {"attack": 0.79, "defense": 0.79},
            "Aston Villa":      {"attack": 0.78, "defense": 0.77},
            "Brighton":         {"attack": 0.74, "defense": 0.72},
            "Nottingham Forest": {"attack": 0.65, "defense": 0.72},
            "Fulham":           {"attack": 0.67, "defense": 0.66},
            "Brentford":        {"attack": 0.68, "defense": 0.67},
            "West Ham":         {"attack": 0.67, "defense": 0.65},
            "Crystal Palace":   {"attack": 0.63, "defense": 0.63},
            "Bournemouth":      {"attack": 0.65, "defense": 0.63},
            "Everton":          {"attack": 0.61, "defense": 0.60},
            "Wolves":           {"attack": 0.63, "defense": 0.62},
            "Leeds United":     {"attack": 0.68, "defense": 0.64},
            "Burnley":          {"attack": 0.60, "defense": 0.59},
            "Sunderland":       {"attack": 0.59, "defense": 0.58},
        }

    def _simulate_match(
        self,
        home_team: str,
        away_team: str,
        match_date: datetime,
        matchday: int,
    ) -> dict:
        """Simulate a single match with realistic statistics."""
        home_str = self.team_strength[home_team]
        away_str = self.team_strength[away_team]

        home_advantage = 0.08  # ~8% boost for home side

        # Expected goals (Poisson-distributed)
        home_xg = 1.5 * (home_str["attack"] + home_advantage) * (1 - away_str["defense"] + 0.3)
        away_xg = 1.2 * away_str["attack"] * (1 - home_str["defense"] + 0.3)

        home_goals = int(self.rng.poisson(max(home_xg, 0.3)))
        away_goals = int(self.rng.poisson(max(away_xg, 0.3)))

        # Shots proportional to xG
        home_shots = max(int(self.rng.normal(home_xg * 7, 3)), 2)
        away_shots = max(int(self.rng.normal(away_xg * 7, 3)), 2)
        home_shots_on = max(int(home_shots * self.rng.uniform(0.3, 0.6)), home_goals)
        away_shots_on = max(int(away_shots * self.rng.uniform(0.3, 0.6)), away_goals)

        # Possession (roughly correlated with strength)
        total_str = home_str["attack"] + away_str["attack"]
        home_poss = round(
            (home_str["attack"] / total_str) * 100 * self.rng.uniform(0.85, 1.15)
        )
        home_poss = max(30, min(70, home_poss))
        away_poss = 100 - home_poss

        # Result label
        if home_goals > away_goals:
            result = "H"
        elif home_goals < away_goals:
            result = "A"
        else:
            result = "D"

        return {
            "match_id": self.rng.integers(100000, 999999),
            "date": match_date.strftime("%Y-%m-%d"),
            "season": "2025-2026",
            "matchday": matchday,
            "home_team": home_team,
            "away_team": away_team,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "home_shots": home_shots,
            "away_shots": away_shots,
            "home_shots_on_target": home_shots_on,
            "away_shots_on_target": away_shots_on,
            "home_possession": home_poss,
            "away_possession": away_poss,
            "result": result,
            "status": "FINISHED",
        }

    def generate_season(self, n_matchdays: int = 30) -> pd.DataFrame:
        """Generate a full (or partial) season of PL matches."""
        records = []
        season_start = datetime(2025, 8, 16)
        teams = list(self.team_strength.keys())

        for matchday in range(1, n_matchdays + 1):
            match_date = season_start + timedelta(weeks=matchday - 1)
            # Round-robin pairs for this matchday (simple rotation)
            pairs = []
            mid = matchday % (len(teams) - 1) + 1
            rotated = teams[:1] + teams[mid:] + teams[1:mid]
            for i in range(len(teams) // 2):
                if matchday % 2 == 0:
                    pairs.append((rotated[i], rotated[-(i + 1)]))
                else:
                    pairs.append((rotated[-(i + 1)], rotated[i]))

            for home, away in pairs:
                records.append(self._simulate_match(home, away, match_date, matchday))

        df = pd.DataFrame(records)
        logger.info(
            "Generated %d synthetic matches across %d matchdays", len(df), n_matchdays
        )
        return df


# ── Orchestrator ───────────────────────────────────────────────────────────────

def collect_data(use_synthetic: bool = False) -> pd.DataFrame:
    """
    Main entry point. Tries the live API first; falls back to synthetic data.

    Parameters
    ----------
    use_synthetic : bool
        Force synthetic data generation (useful for development/testing).

    Returns
    -------
    pd.DataFrame
        Raw match data saved to data/raw/matches_raw.csv
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    df = pd.DataFrame()

    if not use_synthetic:
        api_key = os.getenv("FOOTBALL_DATA_API_KEY", "")
        client = FootballDataClient(api_key=api_key)
        raw = client.get_matches(season="2025")
        if raw:
            df = client.parse_matches(raw)

    if df.empty:
        logger.info("No live data available – generating synthetic dataset.")
        gen = SyntheticDataGenerator(seed=42)
        df = gen.generate_season(n_matchdays=30)

    output_path = RAW_DATA_DIR / "matches_raw.csv"
    df.to_csv(output_path, index=False)
    logger.info("Raw data saved to %s (%d rows)", output_path, len(df))

    # Also save a JSON snapshot
    json_path = RAW_DATA_DIR / "matches_raw.json"
    df.to_json(json_path, orient="records", indent=2)
    logger.info("JSON snapshot saved to %s", json_path)

    return df


if __name__ == "__main__":
    collect_data()
