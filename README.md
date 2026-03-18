<div align="center">

# Premier League Match Predictor

**End-to-end ML system that predicts Premier League 2025-26 match outcomes using XGBoost & Random Forest**

[![CI](https://github.com/sfx1200/football-predict-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/sfx1200/football-predict-ml/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?logo=streamlit&logoColor=white)](https://sfx1200-football-predict-ml.streamlit.app)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](./docker-compose.yml)

### [🚀 Open Live Dashboard →](https://sfx1200-football-predict-ml.streamlit.app)

</div>

---

## What it does

Given any two Premier League clubs, the system returns win/draw/loss probabilities using engineered features like rolling form, head-to-head record, shot conversion rate, and home advantage. Two models are available — XGBoost and Random Forest — both trained with time-series cross-validation to avoid data leakage.

The project covers the complete data science workflow:

```
Raw match data → Cleaning → Feature engineering → Training → REST API + Interactive dashboard
```

---

## Dashboard

Five pages accessible from the sidebar:

| Page | Content |
|------|---------|
| **Overview** | Season stats, fixture counts, home/away win rates, result distribution |
| **Predict** | Pick two teams, pick a model — get probabilities + recent form badges |
| **Team Stats** | Points, wins, clean sheets, goals, form trend, shot conversion per club |
| **Model Insights** | Confusion matrices, feature importance, model comparison charts |
| **Table** | Full standings with Champions League / Europa / relegation zone highlights |

---

## Tech stack

| Layer | Tools |
|-------|-------|
| **ML** | scikit-learn · XGBoost · pandas · numpy |
| **API** | FastAPI · uvicorn · Pydantic |
| **Dashboard** | Streamlit · Plotly |
| **Infra** | Docker · Docker Compose · GitHub Actions |
| **Testing** | pytest · pytest-cov · flake8 · black |

---

## ML features

| Feature | Description |
|---------|-------------|
| `home_form` / `away_form` | Points in last 5 matches |
| `home_avg_scored` / `away_avg_scored` | Rolling average goals scored |
| `home_avg_conceded` / `away_avg_conceded` | Rolling average goals conceded |
| `home_conversion_rate` | Shots on target ÷ shots taken |
| `form_diff` | Rolling form differential (home − away) |
| `h2h_win_rate` | Home team win rate in last 5 H2H meetings |
| `home_goal_diff` | Rolling goal difference |
| `home_advantage` | Binary home-side indicator |

---

## Model performance

| Model | CV accuracy |
|-------|-------------|
| XGBoost | 52 – 58% |
| Random Forest | 50 – 55% |

Football prediction is inherently noisy — top bookmakers sit around 55-60%. Results are in a realistic range given the synthetic training data.

---

## Project structure

```
football_predict_ml/
├── src/
│   ├── data_collection.py      # Generate / fetch match data
│   ├── data_cleaning.py        # Validate and normalise
│   ├── feature_engineering.py  # Rolling stats, H2H, form
│   ├── train_model.py          # Train + hyperparameter search
│   ├── evaluate_model.py       # Metrics and plots
│   └── predict.py              # Inference class
├── api/
│   └── main.py                 # FastAPI: /predict, /predict/batch
├── dashboard/
│   └── app.py                  # Streamlit multi-page dashboard
├── data/processed/             # Cleaned data and feature matrix
├── models/                     # Trained models + evaluation plots
├── tests/                      # pytest suite (data, features, API)
├── streamlit_app.py            # Streamlit Cloud entry point
├── run_pipeline.py             # Run full pipeline in one command
├── docker-compose.yml
└── requirements.txt
```

---

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/sfx1200/football-predict-ml.git
cd football-predict-ml
pip install -r requirements.txt

# 2. Run the full pipeline (data → clean → features → train → evaluate)
python run_pipeline.py

# 3. Dashboard
streamlit run dashboard/app.py
# Open http://localhost:8501

# 4. REST API (separate terminal)
uvicorn api.main:app --reload
# Open http://localhost:8000/docs
```

### Docker

```bash
docker-compose up   # API on :8000, dashboard on :8501
```

---

## REST API

**Predict a match:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team": "Arsenal", "away_team": "Liverpool"}'
```

```json
{
  "home_team": "Arsenal",
  "away_team": "Liverpool",
  "home_win": 0.42,
  "draw": 0.25,
  "away_win": 0.33,
  "predicted_outcome": "Home Win",
  "model_used": "xgboost"
}
```

Batch predictions: `POST /predict/batch` · Interactive docs: `GET /docs`

---

## Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Teams (2025-26)

Arsenal · Aston Villa · Bournemouth · Brentford · Brighton · Burnley · Chelsea · Crystal Palace · Everton · Fulham · Leeds United · Liverpool · Manchester City · Manchester United · Newcastle · Nottingham Forest · Sunderland · Tottenham · West Ham · Wolves
