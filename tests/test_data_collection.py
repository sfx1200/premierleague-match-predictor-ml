"""Tests for data_collection module."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_collection import SyntheticDataGenerator


class TestSyntheticDataGenerator:
    def setup_method(self):
        self.gen = SyntheticDataGenerator(seed=0)

    def test_generate_returns_dataframe(self):
        df = self.gen.generate_season(n_matchdays=3)
        assert isinstance(df, pd.DataFrame)

    def test_correct_number_of_matches(self):
        # 20 teams → 10 matches per matchday
        df = self.gen.generate_season(n_matchdays=5)
        assert len(df) == 50  # 5 × 10

    def test_required_columns(self):
        df = self.gen.generate_season(n_matchdays=2)
        required = [
            "date", "home_team", "away_team",
            "home_goals", "away_goals", "result",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_result_values(self):
        df = self.gen.generate_season(n_matchdays=3)
        assert set(df["result"].unique()).issubset({"H", "D", "A"})

    def test_goals_non_negative(self):
        df = self.gen.generate_season(n_matchdays=3)
        assert (df["home_goals"] >= 0).all()
        assert (df["away_goals"] >= 0).all()

    def test_possession_in_range(self):
        df = self.gen.generate_season(n_matchdays=3)
        assert (df["home_possession"] >= 30).all()
        assert (df["home_possession"] <= 70).all()

    def test_result_consistent_with_goals(self):
        df = self.gen.generate_season(n_matchdays=5)
        home_wins = df[df["result"] == "H"]
        away_wins = df[df["result"] == "A"]
        draws     = df[df["result"] == "D"]
        assert (home_wins["home_goals"] > home_wins["away_goals"]).all()
        assert (away_wins["away_goals"] > away_wins["home_goals"]).all()
        assert (draws["home_goals"] == draws["away_goals"]).all()
