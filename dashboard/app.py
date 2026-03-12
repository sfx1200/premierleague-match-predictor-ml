"""
Streamlit Dashboard
===================
Interactive dashboard for exploring:
  - Team statistics & performance trends
  - Live match predictions
  - Model insights & evaluation metrics
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚽ PL Match Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

PL_TEAMS = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Chelsea", "Crystal Palace", "Everton", "Fulham", "Ipswich Town",
    "Leicester City", "Liverpool", "Manchester City", "Manchester United",
    "Newcastle United", "Nottingham Forest", "Southampton", "Tottenham",
    "West Ham", "Wolves",
]

MODELS_DIR    = Path("models")
PROCESSED_DIR = Path("data/processed")


# ── Data loaders ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_features() -> pd.DataFrame | None:
    path = PROCESSED_DIR / "features.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    return None


@st.cache_resource
def load_predictor(model_name: str):
    try:
        from predict import get_predictor
        return get_predictor(model_name)
    except Exception as exc:
        return None


# ── Helpers ────────────────────────────────────────────────────────────────────

def models_available() -> list[str]:
    available = []
    for name in ["xgboost", "random_forest"]:
        if (MODELS_DIR / f"{name}.joblib").exists():
            available.append(name)
    return available


def league_table(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        ht, at = row["home_team"], row["away_team"]
        hg, ag = row.get("home_goals", 0), row.get("away_goals", 0)
        res = row["result"]
        records.append((ht, hg, ag, "H" if res == "H" else ("D" if res == "D" else "A"), True))
        records.append((at, ag, hg, "A" if res == "A" else ("D" if res == "D" else "H"), False))

    rows = []
    for team, gf, gc, outcome, _ in records:
        rows.append({
            "team": team,
            "gf": gf, "gc": gc,
            "won": 1 if outcome == "H" else 0,
            "drawn": 1 if outcome == "D" else 0,
            "lost": 1 if outcome == "A" else 0,
        })

    table = (
        pd.DataFrame(rows)
        .groupby("team")
        .agg(P=("gf", "count"), W=("won", "sum"), D=("drawn", "sum"),
             L=("lost", "sum"), GF=("gf", "sum"), GC=("gc", "sum"))
        .assign(GD=lambda x: x["GF"] - x["GC"], Pts=lambda x: x["W"] * 3 + x["D"])
        .sort_values("Pts", ascending=False)
        .reset_index()
    )
    table.index = range(1, len(table) + 1)
    return table


# ── Sidebar ────────────────────────────────────────────────────────────────────

st.sidebar.title("⚽ PL Predictor 2025-26")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🔮 Predict Match", "📊 Team Stats", "📈 Model Insights", "🏆 League Table"],
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Home
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("⚽ Premier League 2025-26 Match Predictor")
    st.markdown(
        """
        Welcome to the **Premier League Match Prediction Dashboard**.

        This app uses Machine Learning models (Random Forest & XGBoost) trained on
        historical match data to predict match outcomes.

        ### Features
        - 🔮 **Predict Match** – Get win/draw/loss probabilities for any fixture
        - 📊 **Team Stats** – Explore team performance metrics and trends
        - 📈 **Model Insights** – View feature importances and model accuracy
        - 🏆 **League Table** – Simulated standings from our dataset
        """
    )

    df = load_features()
    avail = models_available()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Matches", len(df) if df is not None else 0)
    with col2:
        st.metric("Teams", len(PL_TEAMS))
    with col3:
        st.metric("Trained Models", len(avail))

    if not avail:
        st.warning(
            "No trained models found. Run the pipeline first:\n"
            "```\npython src/data_collection.py\n"
            "python src/data_cleaning.py\n"
            "python src/feature_engineering.py\n"
            "python src/train_model.py\n```"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Predict Match
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Match":
    st.title("🔮 Match Outcome Predictor")
    st.markdown("Select two teams and a model to generate a prediction.")

    avail = models_available()
    if not avail:
        st.error("No trained models available. Please train the models first.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        home_team = st.selectbox("Home Team", PL_TEAMS, index=0)
    with col2:
        away_options = [t for t in PL_TEAMS if t != home_team]
        away_team = st.selectbox("Away Team", away_options, index=1)
    with col3:
        model_choice = st.selectbox("Model", avail)

    if st.button("⚡ Predict", type="primary"):
        predictor = load_predictor(model_choice)
        if predictor is None:
            st.error("Failed to load model.")
        else:
            with st.spinner("Generating prediction…"):
                result = predictor.predict(home_team, away_team)

            st.success(f"**Predicted Outcome: {result['predicted_outcome']}**")

            # Gauge-style probability bars
            c1, c2, c3 = st.columns(3)
            c1.metric(f"🏠 {home_team} Win", f"{result['home_win']*100:.1f}%")
            c2.metric("🤝 Draw", f"{result['draw']*100:.1f}%")
            c3.metric(f"✈️ {away_team} Win", f"{result['away_win']*100:.1f}%")

            fig = go.Figure(go.Bar(
                x=[f"{home_team} Win", "Draw", f"{away_team} Win"],
                y=[result["home_win"], result["draw"], result["away_win"]],
                marker_color=["steelblue", "orange", "coral"],
                text=[f"{v*100:.1f}%" for v in [result["home_win"], result["draw"], result["away_win"]]],
                textposition="outside",
            ))
            fig.update_layout(
                title=f"{home_team} vs {away_team} – Outcome Probabilities",
                yaxis_range=[0, 1],
                yaxis_tickformat=".0%",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Team Stats
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Team Stats":
    st.title("📊 Team Performance Statistics")
    df = load_features()
    if df is None:
        st.error("Feature data not found. Run the data pipeline first.")
        st.stop()

    selected_team = st.selectbox("Select Team", sorted(PL_TEAMS))

    team_home = df[df["home_team"] == selected_team].copy()
    team_away = df[df["away_team"] == selected_team].copy()

    # Summary stats
    total = len(team_home) + len(team_away)
    home_wins = (team_home["result"] == "H").sum()
    away_wins = (team_away["result"] == "A").sum()
    draws_h   = (team_home["result"] == "D").sum()
    draws_a   = (team_away["result"] == "D").sum()
    total_wins  = home_wins + away_wins
    total_draws = draws_h + draws_a
    total_losses= total - total_wins - total_draws

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Matches Played", total)
    col2.metric("Wins", total_wins)
    col3.metric("Draws", total_draws)
    col4.metric("Losses", total_losses)

    # Form trend
    if not team_home.empty:
        st.subheader("Home Form Trend")
        fig = px.line(
            team_home.sort_values("date"),
            x="date", y="home_form",
            markers=True,
            title=f"{selected_team} – Rolling Form (Home)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Avg goals
    col_l, col_r = st.columns(2)
    with col_l:
        if not team_home.empty:
            fig2 = px.bar(
                team_home.sort_values("date"),
                x="date", y=["home_avg_scored", "home_avg_conceded"],
                barmode="group",
                title=f"{selected_team} – Avg Goals (Home Games)",
            )
            st.plotly_chart(fig2, use_container_width=True)

    with col_r:
        if not team_away.empty:
            fig3 = px.bar(
                team_away.sort_values("date"),
                x="date", y=["away_avg_scored", "away_avg_conceded"],
                barmode="group",
                title=f"{selected_team} – Avg Goals (Away Games)",
            )
            st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Model Insights
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Insights":
    st.title("📈 Model Insights")

    plots_dir = Path("models/plots")

    avail_imgs = list(plots_dir.glob("*.png")) if plots_dir.exists() else []

    if not avail_imgs:
        st.info(
            "No evaluation plots found yet. Run:\n"
            "```\npython src/evaluate_model.py\n```"
        )
    else:
        tabs_map = {}
        for img in sorted(avail_imgs):
            label = img.stem.replace("_", " ").title()
            tabs_map[label] = img

        tab_labels = list(tabs_map.keys())
        tabs = st.tabs(tab_labels)
        for tab, label in zip(tabs, tab_labels):
            with tab:
                st.image(str(tabs_map[label]), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: League Table
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 League Table":
    st.title("🏆 Simulated League Table")
    df = load_features()
    if df is None:
        st.error("No data available.")
        st.stop()

    table = league_table(df)
    st.dataframe(table, use_container_width=True)

    fig = px.bar(
        table.head(10), x="team", y="Pts",
        color="Pts", color_continuous_scale="Blues",
        title="Top 10 Teams by Points",
    )
    st.plotly_chart(fig, use_container_width=True)
