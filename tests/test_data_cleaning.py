"""Tests for data_cleaning module."""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_cleaning import (
    standardise_team_names,
    convert_dates,
    cast_numeric_columns,
    remove_duplicates,
    add_result_column,
    validate_possession,
)


def make_df(**kwargs) -> pd.DataFrame:
    defaults = {
        "date": ["2025-08-16", "2025-08-23"],
        "home_team": ["Arsenal", "Liverpool"],
        "away_team": ["Chelsea", "Man City"],
        "home_goals": [2.0, 1.0],
        "away_goals": [1.0, 1.0],
        "home_possession": [55.0, 48.0],
        "away_possession": [45.0, 52.0],
        "status": ["FINISHED", "FINISHED"],
    }
    defaults.update(kwargs)
    return pd.DataFrame(defaults)


class TestStandardiseTeamNames:
    def test_alias_replaced(self):
        df = make_df(home_team=["Man City", "Man Utd"])
        df = standardise_team_names(df)
        assert df["home_team"].tolist() == ["Manchester City", "Manchester United"]

    def test_unknown_name_unchanged(self):
        df = make_df(home_team=["Arsenal", "Brighton"])
        df = standardise_team_names(df)
        assert df["home_team"].iloc[0] == "Arsenal"


class TestConvertDates:
    def test_parses_iso_dates(self):
        df = make_df()
        df = convert_dates(df)
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_calendar_features_added(self):
        df = make_df()
        df = convert_dates(df)
        for col in ("year", "month", "day_of_week"):
            assert col in df.columns


class TestCastNumericColumns:
    def test_fills_nan_with_median(self):
        df = make_df(home_goals=[2.0, np.nan])
        df = cast_numeric_columns(df)
        assert not df["home_goals"].isna().any()

    def test_string_coerced_to_float(self):
        df = make_df(home_goals=["2", "bad"])
        df = cast_numeric_columns(df)
        assert pd.api.types.is_float_dtype(df["home_goals"])


class TestRemoveDuplicates:
    def test_keeps_first(self):
        df = pd.DataFrame({
            "date": ["2025-08-16", "2025-08-16"],
            "home_team": ["Arsenal", "Arsenal"],
            "away_team": ["Chelsea", "Chelsea"],
            "home_goals": [2, 2],
            "away_goals": [0, 0],
        })
        result = remove_duplicates(df)
        assert len(result) == 1


class TestAddResultColumn:
    def test_home_win(self):
        df = pd.DataFrame({"home_goals": [2], "away_goals": [0]})
        df = add_result_column(df)
        assert df["result"].iloc[0] == "H"

    def test_away_win(self):
        df = pd.DataFrame({"home_goals": [0], "away_goals": [1]})
        df = add_result_column(df)
        assert df["result"].iloc[0] == "A"

    def test_draw(self):
        df = pd.DataFrame({"home_goals": [1], "away_goals": [1]})
        df = add_result_column(df)
        assert df["result"].iloc[0] == "D"


class TestValidatePossession:
    def test_normalises_anomalous_rows(self):
        df = pd.DataFrame({
            "home_possession": [60.0],
            "away_possession": [30.0],  # sum=90, not 100
        })
        result = validate_possession(df)
        total = result["home_possession"] + result["away_possession"]
        assert abs(total.iloc[0] - 100) < 1
