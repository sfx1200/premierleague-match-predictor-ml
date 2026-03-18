"""
Premier League 2025-26 Match Predictor Dashboard
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

st.set_page_config(
    page_title="PL Predictor · 2025-26",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: #0b0f1a;
    color: #dde3f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0e1422;
    border-right: 1px solid #1e2a40;
}
[data-testid="stSidebar"] .stRadio label {
    color: #8897b4 !important;
    font-size: 0.88rem;
    padding: 5px 0;
}
[data-testid="stSidebar"] .stRadio label:hover { color: #c8d4e8 !important; }

#MainMenu, footer, header { visibility: hidden; }

/* Cards */
.stat-card {
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 12px;
    padding: 22px 18px;
    text-align: center;
}
.stat-card .label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #5a7a9e;
    margin-bottom: 10px;
}
.stat-card .val {
    font-size: 2rem;
    font-weight: 800;
    color: #f0f4ff;
    line-height: 1;
}
.stat-card .sub {
    font-size: 0.72rem;
    color: #3d5270;
    margin-top: 8px;
}

/* Section headers */
.sec-head {
    font-size: 1.15rem;
    font-weight: 700;
    color: #e8eeff;
    margin-bottom: 4px;
}
.sec-sub {
    font-size: 0.78rem;
    color: #3d5270;
    margin-bottom: 20px;
}

/* Hero */
.hero {
    background: linear-gradient(120deg, #0f1e35 0%, #0b1525 60%, #0f2040 100%);
    border: 1px solid #1a2d46;
    border-radius: 16px;
    padding: 36px 44px;
    margin-bottom: 28px;
}
.hero-eyebrow {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4a7fb5;
    margin-bottom: 12px;
}
.hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    color: #f0f4ff;
    line-height: 1.15;
    margin-bottom: 12px;
}
.hero-title em {
    font-style: normal;
    color: #5b9bd5;
}
.hero-desc {
    font-size: 0.9rem;
    color: #6b82a0;
    max-width: 520px;
    line-height: 1.65;
}

/* Zone badges */
.zone-cl   { color: #4a9edd; }
.zone-el   { color: #f6c343; }
.zone-conf { color: #78c27a; }
.zone-rel  { color: #e05555; }

/* Predict result */
.pred-box {
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 14px;
    padding: 28px 24px;
    text-align: center;
    margin-top: 16px;
}
.pred-outcome {
    font-size: 1.6rem;
    font-weight: 800;
    margin-bottom: 6px;
}
.pred-meta {
    font-size: 0.78rem;
    color: #3d5270;
}

/* Selectbox */
.stSelectbox > div > div {
    background: #111827 !important;
    border: 1px solid #1e2d45 !important;
    border-radius: 8px !important;
    color: #dde3f0 !important;
}

/* Button */
.stButton > button {
    background: #1a4a7a !important;
    color: #dde3f0 !important;
    border: 1px solid #2a5a8a !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px 28px !important;
    font-size: 0.9rem !important;
    width: 100% !important;
    transition: background 0.2s !important;
}
.stButton > button:hover {
    background: #1f5c9a !important;
}

.divider {
    height: 1px;
    background: #1a2535;
    margin: 28px 0;
}

/* Status pills */
.pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    margin: 2px;
}
.pill-ok  { background: #0f2a1a; color: #4ec97a; border: 1px solid #1a4a2a; }
.pill-err { background: #2a0f0f; color: #e05555; border: 1px solid #4a1a1a; }
</style>
""", unsafe_allow_html=True)

# ── Constants ───────────────────────────────────────────────────────────────────
PL_TEAMS = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Burnley", "Chelsea", "Crystal Palace", "Everton", "Fulham",
    "Leeds United", "Liverpool", "Manchester City", "Manchester United",
    "Newcastle United", "Nottingham Forest", "Sunderland", "Tottenham",
    "West Ham", "Wolves",
]

TEAM_COLORS = {
    "Arsenal":           "#EF0107",
    "Aston Villa":       "#670E36",
    "Bournemouth":       "#DA291C",
    "Brentford":         "#E30613",
    "Brighton":          "#0057B8",
    "Burnley":           "#6C1D45",
    "Chelsea":           "#034694",
    "Crystal Palace":    "#1B458F",
    "Everton":           "#003399",
    "Fulham":            "#000000",
    "Leeds United":      "#FFCD00",
    "Liverpool":         "#C8102E",
    "Manchester City":   "#6CABDD",
    "Manchester United": "#DA291C",
    "Newcastle United":  "#241F20",
    "Nottingham Forest": "#DD0000",
    "Sunderland":        "#EB172B",
    "Tottenham":         "#132257",
    "West Ham":          "#7A263A",
    "Wolves":            "#FDB913",
}

MODELS_DIR = Path("models")
PROCESSED_DIR = Path("data/processed")

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#111827",
    font=dict(family="Inter", color="#8897b4"),
    xaxis=dict(gridcolor="#1a2535", linecolor="#1a2535"),
    yaxis=dict(gridcolor="#1a2535", linecolor="#1a2535"),
    margin=dict(t=36, b=36, l=40, r=20),
)


# ── Data helpers ────────────────────────────────────────────────────────────────
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
            won   = (res == "H" and is_home) or (res == "A" and not is_home)
            drawn = res == "D"
            rows.append({
                "team": team, "gf": gf, "gc": gc,
                "won": int(won), "drawn": int(drawn),
                "lost": int(not won and not drawn),
                "cs": int(gc == 0),
            })
    t = (
        pd.DataFrame(rows).groupby("team")
        .agg(
            P=("gf", "count"), W=("won", "sum"), D=("drawn", "sum"),
            L=("lost", "sum"), GF=("gf", "sum"), GC=("gc", "sum"), CS=("cs", "sum"),
        )
        .assign(GD=lambda x: x.GF - x.GC, Pts=lambda x: x.W * 3 + x.D)
        .sort_values(["Pts", "GD", "GF"], ascending=False)
        .reset_index()
    )
    t.index = range(1, len(t) + 1)
    return t


def team_last_5(df: pd.DataFrame, team: str) -> str:
    matches = df[(df["home_team"] == team) | (df["away_team"] == team)].sort_values("date").tail(5)
    results = []
    for _, r in matches.iterrows():
        is_home = r["home_team"] == team
        if r["result"] == "H":
            results.append("W" if is_home else "L")
        elif r["result"] == "A":
            results.append("L" if is_home else "W")
        else:
            results.append("D")
    color_map = {"W": "#4ec97a", "D": "#f6c343", "L": "#e05555"}
    badges = "".join(
        f"<span style='background:{color_map[r]};color:#0b0f1a;font-size:0.65rem;"
        f"font-weight:700;padding:2px 6px;border-radius:4px;margin:1px;'>{r}</span>"
        for r in results
    )
    return badges


# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 24px 0 20px 0;'>
        <div style='font-size:0.65rem; font-weight:700; letter-spacing:0.15em;
                    text-transform:uppercase; color:#3d5270; margin-bottom:10px;'>
            Premier League
        </div>
        <div style='font-size:1.25rem; font-weight:800; color:#e8eeff; line-height:1.2;'>
            Match Predictor
        </div>
        <div style='font-size:0.75rem; color:#3d5270; margin-top:4px;'>2025 – 26 Season</div>
    </div>
    <div style='height:1px; background:#1a2535; margin-bottom:20px;'></div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["Overview", "Predict", "Team Stats", "Model Insights", "Table"],
        label_visibility="collapsed",
    )

    st.markdown("""
    <div style='height:1px; background:#1a2535; margin: 20px 0 16px 0;'></div>
    <div style='font-size:0.68rem; color:#2a3a50; line-height:1.7;'>
        XGBoost &amp; Random Forest<br>
        scikit-learn · pandas · FastAPI
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">2025 – 26 · Premier League</div>
        <div class="hero-title">Match outcome<br><em>prediction engine</em></div>
        <div class="hero-desc">
            Trained on 300+ simulated fixtures from the 2025-26 season.
            Uses rolling form, head-to-head records, shot conversion, and
            attacking/defensive strength to forecast Home Win · Draw · Away Win.
        </div>
    </div>
    """, unsafe_allow_html=True)

    df = load_features()
    avail = models_available()

    total   = len(df) if df is not None else 0
    hw      = int((df["result"] == "H").sum()) if df is not None else 0
    draws   = int((df["result"] == "D").sum()) if df is not None else 0
    aw      = int((df["result"] == "A").sum()) if df is not None else 0

    c1, c2, c3, c4 = st.columns(4)
    for col, lbl, val, sub, color in [
        (c1, "Fixtures",   total,  "in dataset",              "#5b9bd5"),
        (c2, "Home Wins",  hw,     f"{hw/total*100:.0f}% win rate" if total else "—", "#4ec97a"),
        (c3, "Draws",      draws,  f"{draws/total*100:.0f}% of matches" if total else "—", "#f6c343"),
        (c4, "Away Wins",  aw,     f"{aw/total*100:.0f}% win rate" if total else "—", "#e05555"),
    ]:
        col.markdown(f"""
        <div class="stat-card">
            <div class="label">{lbl}</div>
            <div class="val" style="color:{color};">{val}</div>
            <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    if df is not None:
        col_l, col_r = st.columns([3, 2])

        with col_l:
            st.markdown("<div class='sec-head'>Goals per matchday</div>", unsafe_allow_html=True)
            st.markdown("<div class='sec-sub'>Home and away goals across the season</div>", unsafe_allow_html=True)

            md = df.groupby("matchday").agg(
                home_goals=("home_goals", "sum"),
                away_goals=("away_goals", "sum"),
            ).reset_index()

            fig = go.Figure()
            fig.add_trace(go.Bar(x=md["matchday"], y=md["home_goals"],
                                 name="Home", marker_color="#5b9bd5", opacity=0.9))
            fig.add_trace(go.Bar(x=md["matchday"], y=md["away_goals"],
                                 name="Away", marker_color="#e05555", opacity=0.9))
            fig.update_layout(**PLOTLY_BASE, barmode="group", height=280,
                              legend=dict(orientation="h", y=1.1, x=0, font=dict(size=11)),
                              xaxis_title="Matchday", yaxis_title="Goals")
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown("<div class='sec-head'>Result split</div>", unsafe_allow_html=True)
            st.markdown("<div class='sec-sub'>Distribution across all fixtures</div>", unsafe_allow_html=True)

            fig2 = go.Figure(go.Pie(
                labels=["Home Win", "Draw", "Away Win"],
                values=[hw, draws, aw],
                hole=0.6,
                marker=dict(colors=["#5b9bd5", "#f6c343", "#e05555"],
                            line=dict(color="#0b0f1a", width=3)),
                textinfo="percent",
                textfont=dict(color="#e8eeff", size=12),
            ))
            fig2.update_layout(
                **PLOTLY_BASE, height=280, showlegend=True,
                legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center",
                            font=dict(size=11)),
                annotations=[dict(text=f"{total}<br><span style='font-size:10px'>matches</span>",
                                  x=0.5, y=0.5, font_size=16, font_color="#8897b4",
                                  showarrow=False)],
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-head'>Model status</div>", unsafe_allow_html=True)
    html = ""
    for n in ["xgboost", "random_forest"]:
        label = n.replace("_", " ").title()
        if n in avail:
            html += f"<span class='pill pill-ok'>✓ {label}</span>"
        else:
            html += f"<span class='pill pill-err'>✗ {label}</span>"
    st.markdown(html, unsafe_allow_html=True)
    if not avail:
        st.info("Run `python run_pipeline.py` to train the models.")


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Predict":
    st.markdown("<div class='sec-head' style='font-size:1.4rem;'>Match predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-sub'>Select two teams and a model to get outcome probabilities.</div>", unsafe_allow_html=True)

    avail = models_available()
    if not avail:
        st.error("No trained models found. Run `python run_pipeline.py` first.")
        st.stop()

    col1, col2, col3 = st.columns([5, 5, 3])
    with col1:
        st.markdown("<div style='font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#3d5270;margin-bottom:6px;'>Home team</div>", unsafe_allow_html=True)
        home_team = st.selectbox("Home", PL_TEAMS, index=0, label_visibility="collapsed")
    with col2:
        st.markdown("<div style='font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#3d5270;margin-bottom:6px;'>Away team</div>", unsafe_allow_html=True)
        away_opts = [t for t in PL_TEAMS if t != home_team]
        away_team = st.selectbox("Away", away_opts, index=4, label_visibility="collapsed")
    with col3:
        st.markdown("<div style='font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#3d5270;margin-bottom:6px;'>Model</div>", unsafe_allow_html=True)
        model_choice = st.selectbox("Model", avail, label_visibility="collapsed")

    df = load_features()
    if df is not None:
        h_form = team_last_5(df, home_team)
        a_form = team_last_5(df, away_team)
        fc1, fc2 = st.columns(2)
        with fc1:
            st.markdown(
                f"<div style='font-size:0.7rem;color:#3d5270;margin-top:10px;margin-bottom:4px;'>Last 5 · {home_team}</div>"
                f"<div>{h_form}</div>",
                unsafe_allow_html=True,
            )
        with fc2:
            st.markdown(
                f"<div style='font-size:0.7rem;color:#3d5270;margin-top:10px;margin-bottom:4px;'>Last 5 · {away_team}</div>"
                f"<div>{a_form}</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    predict_btn = st.button("Predict outcome")

    if predict_btn:
        predictor = load_predictor(model_choice)
        if predictor is None:
            st.error("Could not load model.")
        else:
            with st.spinner("Running model..."):
                res = predictor.predict(home_team, away_team)

            hw_p = res["home_win"]
            d_p  = res["draw"]
            aw_p = res["away_win"]
            outcome = res["predicted_outcome"]

            home_c = TEAM_COLORS.get(home_team, "#5b9bd5")
            away_c = TEAM_COLORS.get(away_team, "#e05555")
            outcome_c = home_c if "Home" in outcome else ("#f6c343" if "Draw" in outcome else away_c)

            st.markdown(f"""
            <div class="pred-box">
                <div style='font-size:0.68rem;font-weight:700;letter-spacing:0.12em;
                            text-transform:uppercase;color:#3d5270;margin-bottom:10px;'>
                    Predicted outcome
                </div>
                <div class="pred-outcome" style="color:{outcome_c};">{outcome}</div>
                <div class="pred-meta" style='margin-top:6px;'>
                    {home_team} vs {away_team} &nbsp;·&nbsp; {model_choice.replace("_", " ").title()}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

            # Probability bars (horizontal stacked)
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                y=[""], x=[hw_p * 100], orientation="h", name=f"{home_team}",
                marker_color=home_c, opacity=0.85,
                text=f"{hw_p*100:.1f}%", textposition="inside", insidetextanchor="middle",
                textfont=dict(color="#fff", size=11, family="Inter"),
            ))
            fig_bar.add_trace(go.Bar(
                y=[""], x=[d_p * 100], orientation="h", name="Draw",
                marker_color="#f6c343", opacity=0.85,
                text=f"{d_p*100:.1f}%", textposition="inside", insidetextanchor="middle",
                textfont=dict(color="#0b0f1a", size=11, family="Inter"),
            ))
            fig_bar.add_trace(go.Bar(
                y=[""], x=[aw_p * 100], orientation="h", name=f"{away_team}",
                marker_color=away_c, opacity=0.85,
                text=f"{aw_p*100:.1f}%", textposition="inside", insidetextanchor="middle",
                textfont=dict(color="#fff", size=11, family="Inter"),
            ))
            fig_bar.update_layout(
                barmode="stack", height=72,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(visible=False, range=[0, 100]),
                yaxis=dict(visible=False),
                showlegend=True,
                legend=dict(orientation="h", y=-0.6, x=0.5, xanchor="center",
                            font=dict(color="#8897b4", size=11, family="Inter")),
                margin=dict(t=0, b=44, l=0, r=0),
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Individual probability cards
            g1, g2, g3 = st.columns(3)

            def prob_card(col, label, val, color):
                col.markdown(f"""
                <div class="stat-card" style="border-color:{color}22;">
                    <div class="label">{label}</div>
                    <div class="val" style="color:{color};">{val*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)

            prob_card(g1, f"Home · {home_team}", hw_p, home_c)
            prob_card(g2, "Draw", d_p, "#f6c343")
            prob_card(g3, f"Away · {away_team}", aw_p, away_c)


# ══════════════════════════════════════════════════════════════════════════════
# TEAM STATS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Team Stats":
    st.markdown("<div class='sec-head' style='font-size:1.4rem;'>Team statistics</div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-sub'>Individual team performance across the simulated season.</div>", unsafe_allow_html=True)

    df = load_features()
    if df is None:
        st.error("No data found. Run `python run_pipeline.py` first.")
        st.stop()

    selected = st.selectbox("Select team", sorted(PL_TEAMS), index=0)
    tc = TEAM_COLORS.get(selected, "#5b9bd5")

    th = df[df["home_team"] == selected].copy()
    ta = df[df["away_team"] == selected].copy()
    total  = len(th) + len(ta)
    wins   = int((th["result"] == "H").sum() + (ta["result"] == "A").sum())
    draws  = int((th["result"] == "D").sum() + (ta["result"] == "D").sum())
    losses = total - wins - draws
    pts    = wins * 3 + draws
    gf     = int(th["home_goals"].sum() + ta["away_goals"].sum())
    gc     = int(th["away_goals"].sum() + ta["home_goals"].sum())
    cs     = int((th["away_goals"] == 0).sum() + (ta["home_goals"] == 0).sum())

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    for col, lbl, val, sub, color in [
        (m1, "Points",        pts,    f"{total} played",              tc),
        (m2, "Wins",          wins,   f"{wins/total*100:.0f}% rate",  "#4ec97a"),
        (m3, "Draws",         draws,  f"{draws/total*100:.0f}% rate", "#f6c343"),
        (m4, "Losses",        losses, f"{losses/total*100:.0f}% rate", "#e05555"),
        (m5, "Goals Scored",  gf,     f"{gf/total:.2f} per game",     "#5b9bd5"),
        (m6, "Clean Sheets",  cs,     f"{cs/total*100:.0f}% games",   "#8867c0"),
    ]:
        col.markdown(f"""
        <div class="stat-card">
            <div class="label">{lbl}</div>
            <div class="val" style="color:{color};">{val}</div>
            <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("<div class='sec-head'>Form trend</div>", unsafe_allow_html=True)
        st.markdown("<div class='sec-sub'>Rolling points from last 5 matches</div>", unsafe_allow_html=True)
        form_df = th.sort_values("date") if not th.empty else ta.sort_values("date")
        form_col = "home_form" if not th.empty else "away_form"
        if not form_df.empty and form_col in form_df.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=form_df["date"], y=form_df[form_col],
                mode="lines+markers",
                line=dict(color=tc, width=2),
                marker=dict(size=5, color=tc, line=dict(color="#0b0f1a", width=1.5)),
                fill="tozeroy",
                fillcolor=f"rgba({int(tc[1:3], 16)},{int(tc[3:5], 16)},{int(tc[5:7], 16)},0.08)",
                name="Form",
            ))
            fig.update_layout(**PLOTLY_BASE, height=240, yaxis_title="Pts (last 5)", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("<div class='sec-head'>Goals scored vs conceded</div>", unsafe_allow_html=True)
        st.markdown("<div class='sec-sub'>Rolling 5-match averages (home games)</div>", unsafe_allow_html=True)
        if not th.empty and "home_avg_scored" in th.columns:
            th_s = th.sort_values("date")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=th_s["date"], y=th_s["home_avg_scored"],
                name="Avg scored", line=dict(color="#4ec97a", width=2),
                mode="lines+markers", marker=dict(size=4),
            ))
            fig2.add_trace(go.Scatter(
                x=th_s["date"], y=th_s["home_avg_conceded"],
                name="Avg conceded", line=dict(color="#e05555", width=2),
                mode="lines+markers", marker=dict(size=4),
            ))
            fig2.update_layout(**PLOTLY_BASE, height=240,
                               legend=dict(orientation="h", y=1.1, x=0, font=dict(size=10)),
                               yaxis_title="Goals (rolling avg)")
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("<div class='sec-head'>Result breakdown</div>", unsafe_allow_html=True)
        fig3 = go.Figure(go.Pie(
            labels=["Wins", "Draws", "Losses"],
            values=[wins, draws, losses],
            hole=0.62,
            marker=dict(colors=["#4ec97a", "#f6c343", "#e05555"],
                        line=dict(color="#0b0f1a", width=3)),
            textinfo="percent",
            textfont=dict(color="#e8eeff", size=11),
        ))
        fig3.update_layout(
            **PLOTLY_BASE, height=260, showlegend=True,
            legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center", font=dict(size=10)),
            annotations=[dict(text=f"<b>{pts}</b><br><span style='font-size:9px'>PTS</span>",
                              x=0.5, y=0.5, font_size=18, font_color=tc, showarrow=False)],
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        st.markdown("<div class='sec-head'>Shot conversion rate</div>", unsafe_allow_html=True)
        if not th.empty and "home_conversion_rate" in th.columns:
            conv = th.sort_values("date")
            fig4 = go.Figure(go.Bar(
                x=conv["matchday"], y=conv["home_conversion_rate"],
                marker=dict(
                    color=conv["home_conversion_rate"],
                    colorscale=[[0, "#1a2535"], [0.5, "#2a4a6a"], [1, tc]],
                    showscale=False,
                ),
            ))
            fig4.update_layout(**PLOTLY_BASE, height=260,
                               xaxis_title="Matchday", yaxis_title="Conversion rate")
            st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Insights":
    st.markdown("<div class='sec-head' style='font-size:1.4rem;'>Model insights</div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-sub'>Evaluation metrics, feature importance, and confusion matrices.</div>", unsafe_allow_html=True)

    plots_dir = Path("models/plots")
    imgs = sorted(plots_dir.glob("*.png")) if plots_dir.exists() else []

    if not imgs:
        st.info("No plots found. Run `python src/evaluate_model.py` to generate them.")
        st.stop()

    def show_img(path):
        label = path.stem.replace("_", " ").title()
        st.markdown(f"<div style='font-size:0.72rem;color:#3d5270;font-weight:600;margin-bottom:6px;'>{label}</div>",
                    unsafe_allow_html=True)
        st.image(str(path), use_column_width=True)

    comparison = [p for p in imgs if "comparison" in p.stem]
    importance = [p for p in imgs if "importance" in p.stem]
    confusion  = [p for p in imgs if "confusion" in p.stem]
    other      = [p for p in imgs if p not in comparison + importance + confusion]

    for title, group in [
        ("Overall comparison", comparison),
        ("Feature importance", importance),
        ("Confusion matrices", confusion),
        ("Data visualisations", other),
    ]:
        if group:
            st.markdown(f"<div class='sec-head'>{title}</div>", unsafe_allow_html=True)
            cols = st.columns(max(1, min(len(group), 2)))
            for i, img in enumerate(group):
                with cols[i % len(cols)]:
                    show_img(img)
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LEAGUE TABLE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Table":
    st.markdown("<div class='sec-head' style='font-size:1.4rem;'>League table</div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-sub'>Simulated standings based on generated match data.</div>", unsafe_allow_html=True)

    df = load_features()
    if df is None:
        st.error("No data found. Run `python run_pipeline.py` first.")
        st.stop()

    table = league_table(df)

    # Zone legend
    st.markdown("""
    <div style='display:flex;gap:20px;margin-bottom:16px;font-size:0.72rem;font-weight:600;'>
        <span class='zone-cl'>■ Champions League (1–4)</span>
        <span class='zone-el'>■ Europa League (5)</span>
        <span class='zone-conf'>■ Conference League (6)</span>
        <span class='zone-rel'>■ Relegation (18–20)</span>
    </div>
    """, unsafe_allow_html=True)

    def zone_color(i):
        if i <= 4:
            return "rgba(91,155,213,0.12)"
        if i == 5:
            return "rgba(246,195,67,0.10)"
        if i == 6:
            return "rgba(78,201,122,0.10)"
        if i >= 18:
            return "rgba(224,85,85,0.10)"
        return "rgba(17,24,39,0.9)"

    def font_color(i):
        if i <= 4:
            return "#5b9bd5"
        if i == 5:
            return "#f6c343"
        if i == 6:
            return "#4ec97a"
        if i >= 18:
            return "#e05555"
        return "#dde3f0"

    fill_colors  = [zone_color(i) for i in range(1, len(table) + 1)]
    font_colors  = [font_color(i) for i in range(1, len(table) + 1)]
    gd_vals      = table["GD"].apply(lambda x: f"+{x}" if x > 0 else str(x)).tolist()

    fig = go.Figure(data=[go.Table(
        columnwidth=[36, 170, 44, 44, 44, 44, 52, 52, 52, 52],
        header=dict(
            values=["#", "Club", "P", "W", "D", "L", "GF", "GD", "CS", "Pts"],
            fill_color="#0e1422",
            align=["center", "left"] + ["center"] * 8,
            font=dict(color="#3d5270", size=11, family="Inter"),
            line_color="#1a2535",
            height=34,
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
                gd_vals,
                table["CS"].tolist(),
                table["Pts"].tolist(),
            ],
            fill_color=[fill_colors],
            align=["center", "left"] + ["center"] * 8,
            font=dict(
                color=[font_colors, font_colors] + [["#8897b4"] * len(table)] * 6 + [font_colors],
                size=12,
                family="Inter",
            ),
            line_color="#1a2535",
            height=32,
        ),
    )])
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=0, b=0, l=0, r=0),
        height=740,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-head'>Points — top 10</div>", unsafe_allow_html=True)

    top10  = table.head(10)
    colors = [TEAM_COLORS.get(t, "#5b9bd5") for t in top10["team"]]

    fig2 = go.Figure(go.Bar(
        x=top10["team"],
        y=top10["Pts"],
        marker=dict(color=colors, opacity=0.85, line=dict(color="#0b0f1a", width=1)),
        text=top10["Pts"],
        textposition="outside",
        textfont=dict(color="#8897b4", size=11, family="Inter"),
    ))
    fig2.update_layout(
        **PLOTLY_BASE, height=300,
        yaxis_title="Points",
        xaxis_tickangle=-25,
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)
