"""Tests for feature_engineering module."""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from feature_engineering import compute_team_features, impute_early_season


def make_clean_df(n_matchdays: int = 10) -> pd.DataFrame:
    """Create a minimal clean DataFrame for feature engineering tests."""
    from data_collection import SyntheticDataGenerator
    from data_cleaning import (
        standardise_team_names, convert_dates, cast_numeric_columns,
        filter_finished_matches, add_result_column, remove_duplicates,
    )

    gen = SyntheticDataGenerator(seed=99)
    df = gen.generate_season(n_matchdays=n_matchdays)

    # Minimal cleaning
    df = standardise_team_names(df)
    df = convert_dates(df)
    df = cast_numeric_columns(df)
    df = filter_finished_matches(df)
    df = add_result_column(df)
    df = df.sort_values("date").reset_index(drop=True)
    return df


class TestComputeTeamFeatures:
    def setup_method(self):
        self.df = make_clean_df(n_matchdays=8)
        self.features = compute_team_features(self.df)

    def test_returns_dataframe(self):
        assert isinstance(self.features, pd.DataFrame)

    def test_same_row_count(self):
        assert len(self.features) == len(self.df)

    def test_result_preserved(self):
        assert "result" in self.features.columns
        assert set(self.features["result"].unique()).issubset({"H", "D", "A"})

    def test_home_advantage_column(self):
        assert "home_advantage" in self.features.columns
        assert (self.features["home_advantage"] == 1).all()

    def test_form_diff_column_exists(self):
        assert "form_diff" in self.features.columns

    def test_no_extra_rows(self):
        assert len(self.features) == len(self.df)


class TestImputeEarlySeason:
    def test_no_nans_after_imputation(self):
        df = make_clean_df(n_matchdays=5)
        features = compute_team_features(df)
        imputed = impute_early_season(features)
        numeric = imputed.select_dtypes(include=[np.number])
        assert not numeric.isna().any().any()

    def test_shape_preserved(self):
        df = make_clean_df(n_matchdays=5)
        features = compute_team_features(df)
        imputed = impute_early_season(features)
        assert imputed.shape == features.shape
