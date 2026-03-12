"""
Premier League 2025-26 Match Predictor Dashboard
=================================================
Professional Streamlit dashboard with custom styling,
animated charts, and a modern football-themed UI.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="PL Predictor 2025-26",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts & base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1525 50%, #0a1628 100%);
    color: #e8eaf6;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2e 0%, #0a1628 100%);
    border-right: 1px solid rgba(99, 179, 237, 0.15);
}
[data-testid="stSidebar"] .stRadio label {
    color: #a0b4cc !important;
    font-size: 0.9rem;
    padding: 6px 0;
}
[data-testid="stSidebar"] .stRadio label:hover {
    color: #63b3ed !important;
}

/* ── Hide default header/footer ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Custom metric cards ── */
.metric-card {
    background: linear-gradient(135deg, rgba(99,179,237,0.08) 0%, rgba(99,179,237,0.03) 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}
.metric-card:hover {
    border-color: rgba(99,179,237,0.5);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(99,179,237,0.15);
}
.metric-card .label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #63b3ed;
    margin-bottom: 8px;
}
.metric-card .value {
    font-size: 2.2rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1;
}
.metric-card .sub {
    font-size: 0.75rem;
    color: #6b7a99;
    margin-top: 6px;
}

/* ── Section title ── */
.section-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-sub {
    font-size: 0.85rem;
    color: #6b7a99;
    margin-bottom: 28px;
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1a3a5c 0%, #0d2137 50%, #1a2f4a 100%);
    border: 1px solid rgba(99,179,237,0.25);
    border-radius: 20px;
    padding: 40px 48px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '⚽';
    position: absolute;
    right: 40px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 6rem;
    opacity: 0.08;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.2;
    margin-bottom: 10px;
}
.hero-title span {
    background: linear-gradient(90deg, #63b3ed, #90cdf4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 1rem;
    color: #a0b4cc;
    max-width: 560px;
    line-height: 1.6;
}

/* ── Predict card ── */
.predict-result-card {
    background: linear-gradient(135deg, rgba(99,179,237,0.1) 0%, rgba(144,205,244,0.05) 100%);
    border: 1px solid rgba(99,179,237,0.3);
    border-radius: 16px;
    padding: 28px;
    margin-top: 20px;
    text-align: center;
}
.predict-outcome {
    font-size: 1.8rem;
    font-weight: 800;
    color: #63b3ed;
    margin-bottom: 8px;
}
.predict-confidence {
    font-size: 0.85rem;
    color: #6b7a99;
}

/* ── Team select badge ── */
.team-badge {
    background: rgba(99,179,237,0.08);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    font-weight: 600;
    color: #90cdf4;
    font-size: 0.9rem;
    margin-bottom: 8px;
}

/* ── Table styling ── */
.styled-table {
    background: rgba(13,27,46,0.8);
    border-radius: 12px;
    overflow: hidden;
}

/* ── Selectbox & buttons ── */
.stSelectbox > div > div {
    background: rgba(13,27,46,0.9) !important;
    border: 1px solid rgba(99,179,237,0.3) !important;
    border-radius: 10px !important;
    color: #e8eaf6 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #2b6cb0, #3182ce) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 12px 32px !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #3182ce, #4299e1) !important;
    box-shadow: 0 4px 20px rgba(49,130,206,0.4) !important;
    transform: translateY(-1px) !important;
}

/* ── Divider ── */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,179,237,0.3), transparent);
    margin: 32px 0;
}

/* ── Pills ── */
.pill {
    display: inline-block;
    background: rgba(99,179,237,0.12);
    border: 1px solid rgba(99,179,237,0.25);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    color: #63b3ed;
    margin: 2px;
}
.pill-green {
    background: rgba(72,187,120,0.12);
    border-color: rgba(72,187,120,0.25);
    color: #68d391;
}
.pill-red {
    background: rgba(245,101,101,0.12);
    border-color: rgba(245,101,101,0.25);
    color: #fc8181;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
PL_TEAMS = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Chelsea", "Crystal Palace", "Everton", "Fulham", "Ipswich Town",
    "Leicester City", "Liverpool", "Manchester City", "Manchester United",
    "Newcastle United", "Nottingham Forest", "Southampton", "Tottenham",
    "West Ham", "Wolves",
]

TEAM_COLORS = {
    "Arsenal": "#EF0107", "Liverpool": "#C8102E", "Manchester City": "#6CABDD",
    "Chelsea": "#034694", "Tottenham": "#132257", "Manchester United": "#DA291C",
    "Newcastle United": "#241F20", "Aston Villa": "#95BFE5", "Brighton": "#0057B8",
    "West Ham": "#7A263A", "Wolves": "#FDB913", "Fulham": "#CC0000",
    "Brentford": "#E30613", "Crystal Palace": "#1B458F", "Nottingham Forest": "#DD0000",
    "Bournemouth": "#DA291C", "Everton": "#003399", "Leicester City": "#003090",
    "Southampton": "#D71920", "Ipswich Town": "#0044A9",
}

MODELS_DIR = Path("models")
PROCESSED_DIR = Path("data/processed")


# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_features():
    path = PROCESSED_DIR / "features.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    return None


@st.cache_resource
def load_predictor(model_name: str):
    try:
        from predict import get_predictor
        return get_predictor(model_name)
    except Exception:
        return None


def models_available():
    return [n for n in ["xgboost", "random_forest"] if (MODELS_DIR / f"{n}.joblib").exists()]


def league_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        for team, gf, gc, is_home in [
            (row["home_team"], row["home_goals"], row["away_goals"], True),
            (row["away_team"], row["away_goals"], row["home_goals"], False),
        ]:
            res = row["result"]
            won = (res == "H" and is_home) or (res == "A" and not is_home)
            drawn = res == "D"
            rows.append({"team": team, "gf": gf, "gc": gc,
                         "won": int(won), "drawn": int(drawn), "lost": int(not won and not drawn)})
    t = (
        pd.DataFrame(rows).groupby("team")
        .agg(P=("gf", "count"), W=("won", "sum"), D=("drawn", "sum"),
             L=("lost", "sum"), GF=("gf", "sum"), GC=("gc", "sum"))
        .assign(GD=lambda x: x.GF - x.GC, Pts=lambda x: x.W * 3 + x.D)
        .sort_values(["Pts", "GD"], ascending=False)
        .reset_index()
    )
    t.index = range(1, len(t) + 1)
    return t


# ── Plotly theme ───────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,27,46,0.6)",
    font=dict(family="Inter", color="#a0b4cc"),
    xaxis=dict(gridcolor="rgba(99,179,237,0.08)", linecolor="rgba(99,179,237,0.15)"),
    yaxis=dict(gridcolor="rgba(99,179,237,0.08)", linecolor="rgba(99,179,237,0.15)"),
    margin=dict(t=40, b=40, l=40, r=20),
)


def styled_fig(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 28px 0;'>
        <div style='font-size:2.8rem; margin-bottom:8px;'>⚽</div>
        <div style='font-size:1.1rem; font-weight:800; color:#ffffff; letter-spacing:0.02em;'>
            PL Predictor
        </div>
        <div style='font-size:0.75rem; color:#6b7a99; margin-top:4px;'>2025 – 2026 Season</div>
    </div>
    <div style='height:1px; background:linear-gradient(90deg,transparent,rgba(99,179,237,0.3),transparent); margin-bottom:24px;'></div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠  Overview", "🔮  Predict", "📊  Team Stats", "📈  Model Insights", "🏆  Table"],
        label_visibility="collapsed",
    )

    st.markdown("""
    <div style='height:1px; background:linear-gradient(90deg,transparent,rgba(99,179,237,0.3),transparent); margin: 24px 0 20px 0;'></div>
    <div style='font-size:0.7rem; color:#4a5568; text-align:center; line-height:1.6;'>
        Powered by XGBoost & Random Forest<br>
        <span style='color:#2b6cb0;'>scikit-learn · pandas · FastAPI</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    # Hero
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Premier League<br><span>Match Predictor</span></div>
        <div class="hero-sub">
            Machine learning models trained on 300+ match records to predict
            Home Win, Draw, or Away Win probabilities for the 2025-26 season.
        </div>
    </div>
    """, unsafe_allow_html=True)

    df = load_features()
    avail = models_available()

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    total_matches = len(df) if df is not None else 0
    home_wins = (df["result"] == "H").sum() if df is not None else 0
    draws = (df["result"] == "D").sum() if df is not None else 0
    away_wins = (df["result"] == "A").sum() if df is not None else 0

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Total Matches</div>
            <div class="value">{total_matches}</div>
            <div class="sub">PL 2025-26</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Home Wins</div>
            <div class="value" style="color:#68d391;">{home_wins}</div>
            <div class="sub">{home_wins/total_matches*100:.0f}% of matches</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Draws</div>
            <div class="value" style="color:#f6ad55;">{draws}</div>
            <div class="sub">{draws/total_matches*100:.0f}% of matches</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Away Wins</div>
            <div class="value" style="color:#fc8181;">{away_wins}</div>
            <div class="sub">{away_wins/total_matches*100:.0f}% of matches</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    if df is not None:
        col_l, col_r = st.columns([3, 2])

        with col_l:
            st.markdown("<div class='section-title'>📅 Matches Per Matchday</div>", unsafe_allow_html=True)
            st.markdown("<div class='section-sub'>Goals scored across the season</div>", unsafe_allow_html=True)

            md_goals = (
                df.groupby("matchday")
                .agg(home_goals=("home_goals", "sum"), away_goals=("away_goals", "sum"))
                .reset_index()
            )
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=md_goals["matchday"], y=md_goals["home_goals"],
                name="Home Goals", marker_color="#63b3ed", opacity=0.85,
            ))
            fig.add_trace(go.Bar(
                x=md_goals["matchday"], y=md_goals["away_goals"],
                name="Away Goals", marker_color="#fc8181", opacity=0.85,
            ))
            fig.update_layout(**PLOTLY_LAYOUT, barmode="group",
                              legend=dict(orientation="h", y=1.12, x=0),
                              xaxis_title="Matchday", yaxis_title="Total Goals")
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown("<div class='section-title'>🎯 Result Distribution</div>", unsafe_allow_html=True)
            st.markdown("<div class='section-sub'>Overall outcome split</div>", unsafe_allow_html=True)

            fig2 = go.Figure(go.Pie(
                labels=["Home Win", "Draw", "Away Win"],
                values=[home_wins, draws, away_wins],
                hole=0.55,
                marker=dict(colors=["#63b3ed", "#f6ad55", "#fc8181"],
                            line=dict(color="#0a0e1a", width=3)),
                textinfo="percent+label",
                textfont=dict(color="#ffffff", size=13),
            ))
            fig2.update_layout(**PLOTLY_LAYOUT,
                               showlegend=False,
                               annotations=[dict(text="Results", x=0.5, y=0.5,
                                                 font_size=14, font_color="#a0b4cc",
                                                 showarrow=False)])
            st.plotly_chart(fig2, use_container_width=True)

    # Model status pills
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🤖 Model Status</div>", unsafe_allow_html=True)
    pill_html = ""
    for name in ["xgboost", "random_forest"]:
        if name in avail:
            pill_html += f"<span class='pill pill-green'>✓ {name.replace('_', ' ').title()}</span>"
        else:
            pill_html += f"<span class='pill pill-red'>✗ {name.replace('_', ' ').title()}</span>"
    st.markdown(pill_html, unsafe_allow_html=True)

    if not avail:
        st.warning("No models found. Run `python run_pipeline.py` to train them.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Predict
# ══════════════════════════════════════════════════════════════════════════════
elif "Predict" in page:
    st.markdown("<div class='section-title'>🔮 Match Outcome Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Select two teams to get win/draw/loss probabilities</div>", unsafe_allow_html=True)

    avail = models_available()
    if not avail:
        st.error("No trained models available. Run `python run_pipeline.py` first.")
        st.stop()

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.markdown("<div style='font-size:0.8rem; color:#6b7a99; margin-bottom:6px;'>🏠 HOME TEAM</div>", unsafe_allow_html=True)
        home_team = st.selectbox("Home", PL_TEAMS, index=0, label_visibility="collapsed")
    with col2:
        st.markdown("<div style='font-size:0.8rem; color:#6b7a99; margin-bottom:6px;'>✈️ AWAY TEAM</div>", unsafe_allow_html=True)
        away_opts = [t for t in PL_TEAMS if t != home_team]
        away_team = st.selectbox("Away", away_opts, index=4, label_visibility="collapsed")
    with col3:
        st.markdown("<div style='font-size:0.8rem; color:#6b7a99; margin-bottom:6px;'>🤖 MODEL</div>", unsafe_allow_html=True)
        model_choice = st.selectbox("Model", avail, label_visibility="collapsed")

    st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
    predict_btn = st.button("⚡  Predict Match Outcome", type="primary")

    if predict_btn:
        predictor = load_predictor(model_choice)
        if predictor is None:
            st.error("Failed to load model.")
        else:
            with st.spinner("Analysing team data..."):
                result = predictor.predict(home_team, away_team)

            hw = result["home_win"]
            d = result["draw"]
            aw = result["away_win"]
            outcome = result["predicted_outcome"]

            # Outcome header
            outcome_color = "#63b3ed" if "Home" in outcome else ("#f6ad55" if "Draw" in outcome else "#fc8181")
            st.markdown(f"""
            <div class="predict-result-card">
                <div style='font-size:0.75rem; font-weight:600; letter-spacing:0.1em;
                            text-transform:uppercase; color:#6b7a99; margin-bottom:8px;'>
                    PREDICTED OUTCOME
                </div>
                <div class="predict-outcome" style="color:{outcome_color};">{outcome}</div>
                <div class="predict-confidence">
                    {home_team} vs {away_team} · {model_choice.replace("_"," ").title()}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)

            # Probability gauge charts
            g1, g2, g3 = st.columns(3)

            def make_gauge(val, label, color):
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(val * 100, 1),
                    number=dict(suffix="%", font=dict(size=28, color="#ffffff")),
                    title=dict(text=label, font=dict(size=13, color="#a0b4cc")),
                    gauge=dict(
                        axis=dict(range=[0, 100], tickcolor="#4a5568",
                                  tickfont=dict(color="#4a5568", size=10)),
                        bar=dict(color=color, thickness=0.7),
                        bgcolor="rgba(255,255,255,0.03)",
                        bordercolor="rgba(255,255,255,0)",
                        steps=[
                            dict(range=[0, 33], color="rgba(255,255,255,0.03)"),
                            dict(range=[33, 66], color="rgba(255,255,255,0.02)"),
                        ],
                        threshold=dict(line=dict(color=color, width=3), value=round(val * 100, 1)),
                    ),
                ))
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)",
                                  height=200,
                                  margin=dict(t=40, b=10, l=20, r=20),
                                  font=dict(family="Inter"))
                return fig

            with g1:
                st.plotly_chart(make_gauge(hw, f"🏠 {home_team}", "#63b3ed"), use_container_width=True)
            with g2:
                st.plotly_chart(make_gauge(d, "🤝 Draw", "#f6ad55"), use_container_width=True)
            with g3:
                st.plotly_chart(make_gauge(aw, f"✈️ {away_team}", "#fc8181"), use_container_width=True)

            # Horizontal probability bar
            st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                y=[""], x=[hw * 100], orientation="h",
                name=f"{home_team} Win", marker_color="#63b3ed",
                text=f"  {hw*100:.1f}%", textposition="inside",
                insidetextanchor="middle",
            ))
            fig_bar.add_trace(go.Bar(
                y=[""], x=[d * 100], orientation="h",
                name="Draw", marker_color="#f6ad55",
                text=f"  {d*100:.1f}%", textposition="inside",
                insidetextanchor="middle",
            ))
            fig_bar.add_trace(go.Bar(
                y=[""], x=[aw * 100], orientation="h",
                name=f"{away_team} Win", marker_color="#fc8181",
                text=f"  {aw*100:.1f}%", textposition="inside",
                insidetextanchor="middle",
            ))
            fig_bar.update_layout(
                barmode="stack",
                height=80,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(visible=False, range=[0, 100]),
                yaxis=dict(visible=False),
                showlegend=True,
                legend=dict(orientation="h", y=-0.5, x=0.5, xanchor="center",
                            font=dict(color="#a0b4cc", size=12)),
                margin=dict(t=0, b=40, l=0, r=0),
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Team Stats
# ══════════════════════════════════════════════════════════════════════════════
elif "Team Stats" in page:
    st.markdown("<div class='section-title'>📊 Team Performance</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Deep dive into individual team statistics</div>", unsafe_allow_html=True)

    df = load_features()
    if df is None:
        st.error("No data found. Run the pipeline first.")
        st.stop()

    selected = st.selectbox("Select a team", sorted(PL_TEAMS), index=0)
    team_color = TEAM_COLORS.get(selected, "#63b3ed")

    th = df[df["home_team"] == selected].copy()
    ta = df[df["away_team"] == selected].copy()
    total = len(th) + len(ta)
    wins = (th["result"] == "H").sum() + (ta["result"] == "A").sum()
    draws = (th["result"] == "D").sum() + (ta["result"] == "D").sum()
    losses = total - wins - draws
    pts = wins * 3 + draws

    # Summary metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    for col, label, val, sub, color in [
        (m1, "Points", pts, f"{total} played", "#63b3ed"),
        (m2, "Wins", wins, f"{wins/total*100:.0f}% win rate", "#68d391"),
        (m3, "Draws", draws, f"{draws/total*100:.0f}% draw rate", "#f6ad55"),
        (m4, "Losses", losses, f"{losses/total*100:.0f}% loss rate", "#fc8181"),
        (m5, "Goals", int(th["home_goals"].sum() + ta["away_goals"].sum()), "total scored", team_color),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value" style="color:{color};">{val}</div>
            <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown(f"<div class='section-title' style='font-size:1.1rem;'>📈 Form Trend</div>", unsafe_allow_html=True)
        form_df = th.sort_values("date") if not th.empty else ta.sort_values("date")
        form_col = "home_form" if not th.empty else "away_form"
        if not form_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=form_df["date"], y=form_df[form_col],
                mode="lines+markers",
                line=dict(color=team_color, width=2.5),
                marker=dict(size=6, color=team_color,
                            line=dict(color="#0a0e1a", width=1.5)),
                fill="tozeroy",
                fillcolor=f"rgba{tuple(int(team_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.1,)}",
                name="Form (last 5)",
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=260,
                              yaxis_title="Points", xaxis_title="Date")
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("<div class='section-title' style='font-size:1.1rem;'>⚽ Avg Goals (Home)</div>", unsafe_allow_html=True)
        if not th.empty:
            fig2 = go.Figure()
            th_s = th.sort_values("date")
            fig2.add_trace(go.Scatter(
                x=th_s["date"], y=th_s["home_avg_scored"],
                name="Avg Scored", line=dict(color="#68d391", width=2),
                mode="lines+markers", marker=dict(size=5),
            ))
            fig2.add_trace(go.Scatter(
                x=th_s["date"], y=th_s["home_avg_conceded"],
                name="Avg Conceded", line=dict(color="#fc8181", width=2),
                mode="lines+markers", marker=dict(size=5),
            ))
            fig2.update_layout(**PLOTLY_LAYOUT, height=260,
                               yaxis_title="Goals (rolling avg)",
                               legend=dict(orientation="h", y=1.15, x=0))
            st.plotly_chart(fig2, use_container_width=True)

    # W/D/L donut
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("<div class='section-title' style='font-size:1.1rem;'>🥧 Result Breakdown</div>", unsafe_allow_html=True)
        fig3 = go.Figure(go.Pie(
            labels=["Wins", "Draws", "Losses"],
            values=[wins, draws, losses],
            hole=0.6,
            marker=dict(colors=["#68d391", "#f6ad55", "#fc8181"],
                        line=dict(color="#0a0e1a", width=3)),
            textinfo="percent+label",
            textfont=dict(color="#ffffff", size=12),
        ))
        fig3.update_layout(**PLOTLY_LAYOUT, height=280, showlegend=False,
                           annotations=[dict(text=selected[:3].upper(), x=0.5, y=0.5,
                                             font_size=18, font_color=team_color,
                                             font_family="Inter", showarrow=False)])
        st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        st.markdown("<div class='section-title' style='font-size:1.1rem;'>🎯 Shot Conversion Rate</div>", unsafe_allow_html=True)
        if not th.empty and "home_conversion_rate" in th.columns:
            conv = th.sort_values("date")
            fig4 = go.Figure(go.Bar(
                x=conv["matchday"], y=conv["home_conversion_rate"],
                marker=dict(color=conv["home_conversion_rate"],
                            colorscale=[[0, "#1a3a5c"], [0.5, "#2b6cb0"], [1, "#63b3ed"]],
                            showscale=False),
            ))
            fig4.update_layout(**PLOTLY_LAYOUT, height=280,
                               xaxis_title="Matchday", yaxis_title="Conversion Rate")
            st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Model Insights
# ══════════════════════════════════════════════════════════════════════════════
elif "Model Insights" in page:
    st.markdown("<div class='section-title'>📈 Model Insights</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Evaluation metrics, feature importance, and model comparison</div>", unsafe_allow_html=True)

    plots_dir = Path("models/plots")
    avail_imgs = sorted(plots_dir.glob("*.png")) if plots_dir.exists() else []

    if not avail_imgs:
        st.info("No plots found. Run `python src/evaluate_model.py` first.")
        st.stop()

    # Group plots
    comparison = [p for p in avail_imgs if "comparison" in p.stem]
    importance = [p for p in avail_imgs if "importance" in p.stem]
    confusion  = [p for p in avail_imgs if "confusion" in p.stem]
    other      = [p for p in avail_imgs if p not in comparison + importance + confusion]

    def show_image_card(path):
        label = path.stem.replace("_", " ").title()
        st.markdown(f"<div style='font-size:0.8rem; color:#6b7a99; margin-bottom:8px; font-weight:600;'>{label}</div>", unsafe_allow_html=True)
        st.image(str(path), use_column_width=True)

    if comparison:
        st.markdown("#### Overall Comparison")
        cols = st.columns(len(comparison))
        for col, img in zip(cols, comparison):
            with col:
                show_image_card(img)

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    if importance:
        st.markdown("#### Feature Importance")
        cols = st.columns(len(importance))
        for col, img in zip(cols, importance):
            with col:
                show_image_card(img)

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    if confusion:
        st.markdown("#### Confusion Matrices")
        cols = st.columns(len(confusion))
        for col, img in zip(cols, confusion):
            with col:
                show_image_card(img)

    if other:
        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
        st.markdown("#### Data Visualizations")
        cols = st.columns(min(len(other), 2))
        for i, img in enumerate(other):
            with cols[i % 2]:
                show_image_card(img)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: League Table
# ══════════════════════════════════════════════════════════════════════════════
elif "Table" in page:
    st.markdown("<div class='section-title'>🏆 League Table</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Simulated standings based on the dataset</div>", unsafe_allow_html=True)

    df = load_features()
    if df is None:
        st.error("No data available.")
        st.stop()

    table = league_table(df)

    # Styled table with rank coloring
    top4_color = "rgba(99,179,237,0.15)"
    relegation_color = "rgba(245,101,101,0.1)"

    fig = go.Figure(data=[go.Table(
        columnwidth=[40, 160, 50, 50, 50, 50, 60, 60, 60],
        header=dict(
            values=["#", "Team", "P", "W", "D", "L", "GF", "GD", "Pts"],
            fill_color="rgba(99,179,237,0.15)",
            align="center",
            font=dict(color="#63b3ed", size=12, family="Inter"),
            line_color="rgba(99,179,237,0.1)",
            height=36,
        ),
        cells=dict(
            values=[
                table.index.tolist(),
                table["team"].tolist(),
                table["P"].tolist(),
                table["W"].tolist(),
                table["D"].tolist(),
                table["L"].tolist(),
                table["GF"].tolist(),
                table["GD"].apply(lambda x: f"+{x}" if x > 0 else str(x)).tolist(),
                table["Pts"].tolist(),
            ],
            fill_color=[
                ["rgba(99,179,237,0.12)" if i < 4
                 else "rgba(245,101,101,0.08)" if i >= 17
                 else "rgba(13,27,46,0.6)"
                 for i in range(len(table))]
            ],
            align="center",
            font=dict(color=[
                ["#63b3ed" if i < 4 else "#fc8181" if i >= 17 else "#e8eaf6"
                 for i in range(len(table))]
            ], size=12, family="Inter"),
            line_color="rgba(99,179,237,0.08)",
            height=34,
        ),
    )])
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=0, b=0, l=0, r=0),
        height=760,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Legend
    st.markdown("""
    <div style='display:flex; gap:20px; margin-top:8px; font-size:0.78rem;'>
        <span><span class='pill' style='background:rgba(99,179,237,0.12);'>■</span> Champions League</span>
        <span><span class='pill pill-red'>■</span> Relegation Zone</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    # Points bar chart top 10
    st.markdown("<div class='section-title' style='font-size:1.1rem;'>Top 10 by Points</div>", unsafe_allow_html=True)
    top10 = table.head(10)
    colors = [TEAM_COLORS.get(t, "#63b3ed") for t in top10["team"]]

    fig2 = go.Figure(go.Bar(
        x=top10["team"],
        y=top10["Pts"],
        marker=dict(color=colors, line=dict(color="#0a0e1a", width=1.5)),
        text=top10["Pts"],
        textposition="outside",
        textfont=dict(color="#a0b4cc", size=11),
    ))
    fig2.update_layout(
        **PLOTLY_LAYOUT,
        height=340,
        yaxis_title="Points",
        xaxis_tickangle=-30,
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)
