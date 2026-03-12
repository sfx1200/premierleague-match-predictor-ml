"""
Visualization Module
====================
Standalone visualization utilities for team performance trends,
feature distributions, and prediction probabilities.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
PLOTS_DIR     = Path("models/plots")

sns.set_theme(style="whitegrid", palette="tab10")


def plot_team_form_trend(df: pd.DataFrame, teams: list[str] | None = None):
    """Line chart: rolling 5-match form points over the season."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if teams is None:
        # Pick top 6 by final form
        teams = (
            df.sort_values("date").groupby("home_team")["home_form"]
            .last()
            .sort_values(ascending=False)
            .head(6)
            .index.tolist()
        )

    fig, ax = plt.subplots(figsize=(12, 6))
    for team in teams:
        team_df = df[df["home_team"] == team].sort_values("date")
        ax.plot(team_df["date"], team_df["home_form"], marker="o", markersize=4, label=team)

    ax.set_title("Rolling 5-Match Form Points (Home Games)", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Points (last 5 matches)")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    path = PLOTS_DIR / "team_form_trend.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved team form trend to %s", path)


def plot_goals_distribution(df: pd.DataFrame):
    """Distribution of home vs away goals."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, col, color, label in zip(
        axes,
        ["home_goals", "away_goals"],
        ["steelblue", "coral"],
        ["Home Goals", "Away Goals"],
    ):
        sns.histplot(df[col].dropna(), bins=range(0, 10), ax=ax, color=color, stat="density")
        ax.set_title(f"Distribution of {label}", fontsize=12)
        ax.set_xlabel("Goals")

    plt.suptitle("Goals Scored Distribution – PL 2025/26", fontsize=14)
    plt.tight_layout()
    path = PLOTS_DIR / "goals_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved goals distribution to %s", path)


def plot_result_share(df: pd.DataFrame):
    """Pie chart: overall result distribution."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    counts = df["result"].value_counts()
    labels = {"H": "Home Win", "D": "Draw", "A": "Away Win"}

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(
        counts.values,
        labels=[labels.get(k, k) for k in counts.index],
        autopct="%1.1f%%",
        colors=["steelblue", "orange", "coral"],
        startangle=90,
    )
    ax.set_title("Match Result Distribution – PL 2025/26", fontsize=14)
    plt.tight_layout()
    path = PLOTS_DIR / "result_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved result distribution to %s", path)


def plot_correlation_heatmap(df: pd.DataFrame):
    """Heatmap of feature correlations."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    numeric = df.select_dtypes(include=["number"]).drop(
        columns=["match_id", "matchday", "home_goals", "away_goals"], errors="ignore"
    )
    corr = numeric.corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = pd.DataFrame(False, index=corr.index, columns=corr.columns)
    # Upper triangle mask
    import numpy as np
    mask.values[np.triu_indices_from(mask, k=1)] = True

    sns.heatmap(
        corr, mask=mask, annot=False, cmap="RdBu_r",
        center=0, linewidths=0.3, ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14)
    plt.tight_layout()
    path = PLOTS_DIR / "correlation_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved correlation heatmap to %s", path)


def generate_all_visuals():
    """Run all visualizations from the processed feature dataset."""
    df = pd.read_csv(PROCESSED_DIR / "features.csv", parse_dates=["date"])
    plot_team_form_trend(df)
    plot_goals_distribution(df)
    plot_result_share(df)
    plot_correlation_heatmap(df)
    logger.info("All visualizations generated in %s/", PLOTS_DIR)


if __name__ == "__main__":
    generate_all_visuals()
